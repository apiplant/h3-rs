"""Reference dumps (ComfyUI) for the text encoder and both VAEs. H3_WHAT=te,vae,audio,enc"""
import os, sys
sys.path.insert(0, "/mnt/extra/ai/ComfyUI")
os.chdir("/mnt/extra/ai/ComfyUI")
sys.argv = sys.argv[:1] + ["--disable-all-custom-nodes"]
import torch, numpy as np
from PIL import Image
from safetensors.torch import save_file
import comfy.options
comfy.options.enable_args_parsing()
import comfy.sd, comfy.utils, comfy.model_management as mm
torch.set_grad_enabled(False)
torch.inference_mode().__enter__()

out = os.environ["H3_REF"]
M = "/mnt/extra/ai/ComfyUI/models"
what = os.environ.get("H3_WHAT", "te,vae,audio,enc").split(",")
PROMPT = "A red fox trots through fresh snow at dawn, breath steaming, soft wind and crunching footsteps."
IMG = "/mnt/extra/ai/ComfyUI/input/01.png"

def load_img(w, h):
    im = np.asarray(Image.open(IMG).convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(im)[None]
    return comfy.utils.common_upscale(t.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1)

if "te" in what:
    clip = comfy.sd.load_clip([f"{M}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"], clip_type=comfy.sd.CLIPType.MINIMAX)
    d = {}
    for name, images in (("t2v", []), ("i2v", [load_img(384, 320)])):
        tokens = clip.tokenize(PROMPT, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        c, extra = cond[0][0], cond[0][1]
        d[f"{name}_hidden"] = c[0].float().cpu().contiguous()
        d[f"{name}_tags"] = extra["minimax_token_tags"].to(torch.int64).cpu().contiguous()
        if images:
            d[f"{name}_image"] = images[0][0].contiguous()
        ids = [t[0] for t in tokens["qwen3vl_32b"][0] if isinstance(t[0], int)]
        d[f"{name}_ids"] = torch.tensor(ids, dtype=torch.int64)
        print(name, c.shape, extra["minimax_token_tags"].shape)
    save_file(d, f"{out}/te.safetensors")

if "vae" in what or "enc" in what:
    sd, meta = comfy.utils.load_torch_file(f"{M}/vae/minimax_h3_video_vae_fp16.safetensors", return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=meta)
    d = {}
    if "vae" in what:
        g = torch.Generator("cpu").manual_seed(0)
        z = torch.randn(1, 24, 7, 20, 24, generator=g) * 0.8
        px = vae.decode(z)  # [B, T, H, W, 3]
        d.update({"z": z, "pixels": px.float().cpu().contiguous()})
        print("vae decode", px.shape)
    if "enc" in what:
        img = load_img(384, 320)
        lat = vae.encode(img[:, :, :, :3])
        d.update({"image": img[0].contiguous(), "latent": lat.float().cpu().contiguous()})
        print("vae encode", lat.shape)
    save_file(d, f"{out}/vae_{'_'.join(w for w in what if w in ('vae','enc'))}.safetensors")

if "audio" in what:
    sd, meta = comfy.utils.load_torch_file(f"{M}/vae/minimax_h3_audio_vae_fp32.safetensors", return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=meta)
    g = torch.Generator("cpu").manual_seed(1)
    z = torch.randn(1, 32, 2, 16, generator=g) * 0.8
    with torch.no_grad():
        wave = vae.first_stage_model.to(mm.get_torch_device()).decode(z.to(mm.get_torch_device()))
    save_file({"z": z, "wave": wave[0].float().cpu().contiguous()}, f"{out}/audio.safetensors")
    print("audio", wave.shape)

if "venc" in what:
    sd, meta = comfy.utils.load_torch_file(f"{M}/vae/minimax_h3_video_vae_fp16.safetensors", return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=meta)
    g = torch.Generator("cpu").manual_seed(2)
    base = load_img(384, 320)[0]
    frames = torch.stack([torch.roll(base, shifts=4 * i, dims=1) for i in range(22)])  # panning clip
    lat = vae.encode(frames)
    save_file({"frames": frames.contiguous(), "latent": lat.float().cpu().contiguous()}, f"{out}/venc.safetensors")
    print("video encode", lat.shape)

if "aenc" in what:
    sd, meta = comfy.utils.load_torch_file(f"{M}/vae/minimax_h3_audio_vae_fp32.safetensors", return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=meta)
    t = torch.arange(32000 * 2) / 32000.0
    wave = torch.stack([0.3 * torch.sin(2 * 3.14159 * 440 * t) * torch.exp(-t), 0.2 * torch.sin(2 * 3.14159 * 660 * t)])  # [2, L]
    z = vae.encode(wave[None].movedim(1, -1))
    save_file({"wave": wave.contiguous(), "z": z.float().cpu().contiguous()}, f"{out}/aenc.safetensors")
    print("audio encode", z.shape)
