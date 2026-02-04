import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import (
    denoise_fireflow, denoise_rf_solver, denoise, edit_uniedit,
    get_schedule, prepare, unpack, get_noise, latents_shift
)
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5, save_velocity_distribution, get_word_index)
from transformers import pipeline
from PIL import Image
import numpy as np

import os

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    torch.set_grad_enabled(False)
    name = args.name
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    guidance = args.guidance
    output_dir = args.output_dir
    num_steps = args.num_steps
    offload = args.offload
    prefix = args.sampling_solver
    inject = args.inject
    ls_ratio = args.ls_ratio
    edit_object = args.edit_object
    edit_type = args.edit_type
    seed = args.seed if args.seed > 0 else None

    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    # init all components
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    if offload:
        model.cpu()
        torch.cuda.empty_cache()
        ae.encoder.to(torch_device)
   
    init_image = None
    init_image_array = np.array(Image.open(args.source_img_dir).convert('RGB'))
    shape = init_image_array.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
    latent_h = new_h // 16
    latent_w = new_w // 16

    init_image = init_image_array[:new_h, :new_w, :]
    width, height = init_image.shape[0], init_image.shape[1]
    
    t0 = time.perf_counter()
    
    init_image = encode(init_image, torch_device, ae)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")

        # prepare random noise
        z_random = get_noise(
                        1,
                        opts.height,
                        opts.width,
                        device=torch_device,
                        dtype=torch.bfloat16,
                        seed=opts.seed,
                    )

        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)

        
        key_word_index = get_word_index(opts.source_prompt,edit_object,t5)
        info = {}
        info['feature_path'] = args.feature_path
        info['feature'] = {}
        info['inject_step'] = inject
        info['sampling_solver']= args.sampling_solver
        info['latent_h'] = latent_h
        info['latent_w'] = latent_w
        
        # UniEdit-Flow
        info['alpha'] = args.alpha
        info['omega'] = args.omega

        # ProEdit
        info['key_word_index'] = key_word_index
        info['edit_type'] = edit_type
        info['kv_mix_ratio'] = args.kv_mix_ratio
        info['indices'] = []
        info['kv_mix'] = True
        info['kv_mask'] = True
        
        if args.sampling_solver == 'uniedit':
            prefix += '_alpha_%.02f' % args.alpha
            prefix += '_omega_%.02f' % args.omega
        
        prefix += '_inject_' + str(inject)
        prefix += '_editobj_' + str(edit_object)
        prefix += '_kvmix_' + str(args.kv_mix_ratio)
        prefix += '_ls_' + str(ls_ratio)
        
        if not os.path.exists(args.feature_path):
            os.mkdir(args.feature_path)

        if args.sampling_solver == 'uniedit':
            inp = prepare(t5, clip, init_image, prompt="")
            inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)

            # Prepare for Latents-Shift
            inp_random = prepare(t5, clip, z_random, prompt=opts.target_prompt)
            # src cond
            src_tmp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
            inp['src_txt'] = src_tmp['txt']
            inp['src_txt_ids'] = src_tmp['txt_ids']
            inp['src_vec'] = src_tmp['vec']
            inp_target['src_txt'] = src_tmp['txt']
            inp_target['src_txt_ids'] = src_tmp['txt_ids']
            inp_target['src_vec'] = src_tmp['vec']
        else:
            inp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(t5, clip, init_image, prompt=opts.target_prompt)

            # Prepare for Latents-Shift
            inp_random = prepare(t5, clip, z_random, prompt=opts.target_prompt)

        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
        info['mask_time'] = timesteps[-2]

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)
        
        denoise_strategies = {
            'reflow' : denoise,
            'rf_solver' : denoise_rf_solver,
            'fireflow' : denoise_fireflow,
            'uniedit': edit_uniedit,
        }
        if args.sampling_solver not in denoise_strategies:
            raise NotImplementedError("Unknown denoising strategy")
        denoise_strategy = denoise_strategies[args.sampling_solver]

        # inversion initial noise
        z, info = denoise_strategy(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info) 
        
        # Latents-Shift
        if edit_type != 'style':
            z = latents_shift(z, inp_random["img"], info['indices'], alpha = ls_ratio)
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise_strategy(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)
        
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        batch_x = unpack(x.float(), opts.width, opts.height)

        for x in batch_x:
            x = x.unsqueeze(0)
            output_name = os.path.join(output_dir, prefix + "_img_{idx}.jpg")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                idx = 0
            else:
                fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
                if len(fns) > 0:
                    idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
                else:
                    idx = 0

            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            fn = output_name.format(idx=idx)
            print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
            # bring into PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x[0], "c h w -> h w c")

            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
            nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
            
            if nsfw_score < NSFW_THRESHOLD:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = name
                if add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = source_prompt
                img.save(fn, exif=exif_data, quality=95, subsampling=0)
                idx += 1
            else:
                print("Your generated image may contain NSFW content.")

            opts = None

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    parser.add_argument('--source_img_dir', default='', type=str,
                        help='The path of the source image')
    parser.add_argument('--source_prompt', type=str,
                        help='describe the content of the source image (or leaves it as null)')
    parser.add_argument('--target_prompt', type=str,
                        help='describe the requirement of editing')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=1,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=15,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=0,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='the path of the edited image')
    parser.add_argument('--sampling_solver', default='rf_solver', type=str,
                        help='method used to conduct sampling at inference time')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    
    # UniEdit-Flow
    parser.add_argument('--alpha', type=float, default=0.6, help='delay rate of UniEdit-Flow')
    parser.add_argument('--omega', type=float, default=5, help='guidance strength of UniEdit-Flow')

    # ProEdit
    parser.add_argument('--edit_object', type=str, help='set it to control the masked region')
    parser.add_argument('--edit_type', type=str, help='set it to control the editing type: add, remove, change, style')
    parser.add_argument('--kv_mix_ratio', type=float, default=0.9, help='set it to control the KV-Mix ratio during editing')
    parser.add_argument('--ls_ratio', type=float, default=0.25, help='set it to control the Latents-Shift ratio during editing')
    
    args = parser.parse_args()
    
    main(args)