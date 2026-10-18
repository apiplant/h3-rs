"""Reference dump for the latent upscaler: runs LBH-123-AI's `LatentResizer3D` node code (pytorch, fp16 on
cuda, like its default node settings) on fixed random latents. H3_REF=<dir> [H3_NODE=<custom node checkout>]"""
import os, sys, types, importlib.util
import torch
from safetensors.torch import load_file, save_file

node = os.environ.get("H3_NODE", "/mnt/extra/ai/ComfyUI/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler")
out = os.environ["H3_REF"]
M = "/mnt/extra/ai/ComfyUI/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"

# the node module imports ComfyUI's folder_paths at import time; only the network is needed here
fp = types.ModuleType("folder_paths")
fp.folder_names_and_paths = {"latent_upscale_models": ([], set())}
fp.models_dir = "/tmp"
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda n: ["/tmp"]
fp.get_filename_list = lambda n: []
sys.modules["folder_paths"] = fp
spec = importlib.util.spec_from_file_location("mmh3_3d", f"{node}/nodes/minimax_h3_latent_upscaler_3d.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

torch.set_grad_enabled(False)
sd = mod._extract_upscaler_sd(mod._load_raw_sd(M))
cfg = mod._detect_arch(sd)
net = mod.LatentResizer3D(**cfg)
net.load_state_dict(sd, strict=True)
net.eval()  # the node does; dropout would otherwise be active
dev = "cuda"
mean, std = mod._make_norm_tensors(dev, torch.float32)

# (name, T, H, W, Ho, Wo): single segment, chunked (T > 32), odd sizes with a non-integer scale
cases = [("a", 6, 12, 20, 24, 40), ("b", 40, 6, 8, 12, 16), ("c", 5, 10, 14, 17, 22)]
res = {}
g = torch.Generator().manual_seed(7)
for name, t, h, w, ho, wo in cases:
    # latents roughly distributed like the VAE's (per-channel mean/std), not the unit-normal the net sees
    z = (torch.randn(1, 24, t, h, w, generator=g).to(dev) * std + mean)
    scale = (wo / w + ho / h) / 2
    res[f"{name}_z"] = z.cpu().contiguous()
    res[f"{name}_scale"] = torch.tensor([scale])
    for prec, dt in (("f16", torch.float16), ("f32", torch.float32)):
        m = net.to(dev).to(dt)
        x = ((z.to(dt) - mean.to(dt)) / std.to(dt))
        y = m(x, scale=scale, target_size=(t, ho, wo), enable_chunking=True)
        res[f"{name}_{prec}"] = (y.float() * std + mean).cpu().contiguous()
os.makedirs(out, exist_ok=True)
save_file(res, f"{out}/upscaler.safetensors")
print("wrote", f"{out}/upscaler.safetensors", {k: tuple(v.shape) for k, v in res.items()})

# stage-by-stage dump (fp32, case "s": smooth latent) to localise any divergence (parity: upscaler_stages)
t, h, w, ho, wo = 5, 10, 14, 17, 22
base = torch.randn(1, 24, t, 3, 4, generator=g)
z = torch.nn.functional.interpolate(base, size=(t, h, w), mode="trilinear", align_corners=False).to(dev) * std + mean
scale = (wo / w + ho / h) / 2
m = net.to(dev).float()
x = (z - mean) / std
st = {"s_z": z.cpu(), "s_scale": torch.tensor([scale]), "s_x": x.cpu()}
emb = m.embed(torch.tensor([[scale - 1]], device=dev))
h_ = m.conv_in(x); st["s_conv_in"] = h_.cpu()
for i, b in enumerate(m.in_blocks):
    h_ = b(h_, emb.expand(1, -1)) if isinstance(b, mod.ResBlockEmb3D) else b(h_)
    if i < 3: st[f"s_in{i}"] = h_.cpu()
st["s_in_all"] = h_.cpu()
h_ = torch.nn.functional.interpolate(h_, size=(t, ho, wo), mode="trilinear", align_corners=False); st["s_interp"] = h_.cpu()
for i, b in enumerate(m.out_blocks):
    h_ = b(h_, emb.expand(1, -1)) if isinstance(b, mod.ResBlockEmb3D) else b(h_)
st["s_out_all"] = h_.cpu()
h_ = m.conv_out(torch.nn.functional.silu(m.norm_out(h_))); st["s_final"] = (h_ * std + mean).cpu()
for k in list(st): st[k] = st[k].contiguous()
save_file(st, f"{out}/upscaler_stages.safetensors")
print("stages", {k: tuple(v.shape) for k, v in st.items()})
