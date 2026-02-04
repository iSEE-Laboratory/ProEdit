import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image


import torch
import gradio as gr

from flux.sampling import denoise, denoise_fireflow, denoise_rf_solver, edit_uniedit, get_schedule, prepare, unpack, get_noise, latents_shift
from flux.util import (configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5, get_word_index)

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
    with torch.no_grad():
        init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


class FluxEditor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload
        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.output_dir = 'result'
        self.add_sampling_metadata = True

        if self.name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {self.name}, chose from {available}")

        # init all components
        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(self.name, device="cpu" if self.offload else self.device)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)
        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
    
    @torch.inference_mode()
    def edit(
        self, init_image, source_prompt, target_prompt, sampling_solver,
        alpha, omega,
        num_steps, guidance, inject_step,
        edit_type, edit_object, kv_mix_ratio, ls_ratio 
    ):
        torch.cuda.empty_cache()
        seed = None
        zero_init = False
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)
        
        shape = init_image.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        latent_h = new_h // 16
        latent_w = new_w // 16
        init_image = init_image[:new_h, :new_w, :]
        width, height = init_image.shape[0], init_image.shape[1]
        init_image = encode(init_image, self.device, self.ae)


        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        t0 = time.perf_counter()

        z_random = get_noise(
                        1,
                        opts.height,
                        opts.width,
                        device=self.device,
                        dtype=torch.bfloat16,
                        seed=opts.seed,
                    )

        opts.seed = None
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        key_word_index = get_word_index(opts.source_prompt,edit_object,self.t5)
        info = {}
        info['inject_step'] = inject_step
        info['sampling_solver']= sampling_solver
        info['latent_h'] = latent_h
        info['latent_w'] = latent_w
        info['feature'] = {}
        
        # UniEdit-Flow
        info['alpha'] = alpha
        info['omega'] = omega
        info['zero_init'] = zero_init

        # ProEdit
        info['key_word_index'] = key_word_index
        info['edit_type'] = edit_type
        info['edit_object'] = edit_object
        info['kv_mix_ratio'] = kv_mix_ratio
        info['ls_ratio'] = ls_ratio
        info['indices'] = []
        info['kv_mix'] = True
        info['kv_mask'] = True

        if sampling_solver == 'uniedit':
            inp = prepare(self.t5, self.clip, init_image, prompt="")
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)

            # Prepare for Latents-Shift
            inp_random = prepare(self.t5, self.clip, z_random, prompt=opts.target_prompt)
            # src cond
            src_tmp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            inp['src_txt'] = src_tmp['txt']
            inp['src_txt_ids'] = src_tmp['txt_ids']
            inp['src_vec'] = src_tmp['vec']
            inp_target['src_txt'] = src_tmp['txt']
            inp_target['src_txt_ids'] = src_tmp['txt_ids']
            inp_target['src_vec'] = src_tmp['vec']
        else:
            inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)

            # Prepare for Latents-Shift
            inp_random = prepare(self.t5, self.clip, z_random, prompt=opts.target_prompt)
        
        
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        denoise_strategies = {
            'reflow' : denoise,
            'rf_solver' : denoise_rf_solver,
            'fireflow' : denoise_fireflow,
            'uniedit': edit_uniedit,
        }
        denoise_strategy = denoise_strategies[sampling_solver]

        # inversion initial noise
        z, info = denoise_strategy(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)

        # Latents-Shift
        if edit_type != 'style':
            z = latents_shift(z, inp_random["img"], info['indices'], alpha = ls_ratio)
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))

        # denoise initial noise
        x, _ = denoise_strategy(self.model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.width, opts.height)

        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

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
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        if self.add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = source_prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)

        print("End Edit")
        return img



def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    editor = FluxEditor(args)
    is_schnell = model_name == "flux-schnell"

    description = r"""
        <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/iSEE-Laboratory/ProEdit' target='_blank'><b>ProEdit: Inversion-based Editing From Prompts Done Right</b></a>.<br>
    
        💫💫 <b>Here is editing steps:</b> <br>
        1️⃣ Upload your image that needs to be edited. <br>
        2️⃣ Fill in your source prompt, target prompt,sampling solver, edit type and edit object.(The 'edit object' should be selected directly from the source prompt.) <br>
        3️⃣ Adjust the hyperparameters.(Delay rate (α) and Guidance strength (ω) are specifically required for the 'uniedit' sampling solver only.)  <br>
        4️⃣ Click the "Generate" button to generate your edited image! <br>
        
        """
    article = r"""
    If our work is helpful, please help to ⭐ the <a href='https://github.com/iSEE-Laboratory/ProEdit' target='_blank'>Github Repo</a>. Thanks! 
    """
    
    # Pre-defined examples
    examples = [
        ["examples/source/cat.jpg", "A cat wearing a chef hat and a white chef coat, standing in a kitchen and chopping broccoli on a wooden cutting board.", "A dog wearing a chef hat and a white chef coat, standing in a kitchen and chopping broccoli on a wooden cutting board.", "fireflow", 0.8, 5, 15, 2, 5, 'change', 'cat', 0.9, 0.25],
        ["examples/source/hat.jpg", "a girl with a red hat and red t-shirt is sitting in a park, best quality", "a girl with a yellow hat and red t-shirt is sitting in a park, best quality", "uniedit", 0.8, 5, 15, 2, 5, 'change', 'hat', 0.9, 0.25],
        ["examples/source/Venice.jpg", "A young woman is holding a wooden sign with the words ' Venice is awesome ' written in elegant cursive.", "A young woman is holding a wooden sign with the words ' ProEdit is awesome ' written in elegant cursive.", "fireflow", 0.8, 5, 15, 2, 5, 'change', 'Venice', 0.9, 0.25],
    ]

    with gr.Blocks() as demo:
        gr.Markdown(f"# ProEdit Demo (FLUX for image editing)")
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value="Describe the content of the uploaded image.")
                target_prompt = gr.Textbox(label="Target Prompt", value="Describe the desired content of the edited image.")
                edit_object = gr.Textbox(label="Edit Object", value="Specify the object to edit.")
                sampling_solver = gr.Dropdown(choices=['uniedit', 'reflow', 'rf_solver', 'fireflow'], value='fireflow', label="Sampling Solver")
                edit_type = gr.Dropdown(choices=['change', 'add', 'remove', 'style'], value='change', label="Edit Type")
                init_image = gr.Image(label="Input Image", visible=True)
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):
                    alpha = gr.Slider(0.0, 1.0, 0.8, step=0.05, label=f"Delay rate (α)")
                    omega = gr.Slider(2.0, 10.0, 5, step=0.5, label=f"Guidance strength (ω)")
                    num_steps = gr.Slider(1, 50, 15, step=1, label="Number of steps")
                    guidance = gr.Slider(1.0, 2.0, 2, step=0.5, label="CFG Guidance (Optional)", interactive=not is_schnell)
                    
                    # ProEdit options
                    inject_step = gr.Slider(0, 50, 5, step=1, label="Injection Step for ProEdit")
                    kv_mix_ratio = gr.Slider(0.0, 1.0, 0.9, step=0.05, label="Key-Value Mix Ratio for ProEdit")
                    ls_ratio = gr.Slider(0.0, 1.0, 0.25, step=0.05, label="Latents-Shift Ratio for ProEdit")
                output_image = gr.Image(label="Generated Image")
                gr.Markdown(article)

        generate_btn.click(
            fn=editor.edit,
            inputs=[init_image, source_prompt, target_prompt, sampling_solver, alpha, omega, num_steps, guidance, inject_step, edit_type, edit_object, kv_mix_ratio, ls_ratio],
            outputs=[output_image]
        )
        
        # Add examples
        gr.Examples(
            examples=examples,
            inputs=[
                init_image, 
                source_prompt, 
                target_prompt, 
                sampling_solver,
                alpha, 
                omega, 
                num_steps, 
                guidance, 
                inject_step,
                edit_type, 
                edit_object, 
                kv_mix_ratio, 
                ls_ratio
            ],
            outputs=[output_image],
            fn=editor.edit,
        )


    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port", type=int, default=41035)
    args = parser.parse_args()

    demo = create_demo(args.name, args.device, args.offload)
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
