"""Reference dump from ComfyUI for the h3-rs denoise-mask parity test (CPU only).

usage: python tools/ref_inpaint.py <out.safetensors>
Runs ComfyUI's own `reshape_mask`, `MiniMaxH3._token_grid_masks` and `MiniMaxH3.scale_latent_inpaint` on a small
random masked AV latent and writes inputs and outputs; `cargo test --lib inpaint::tests::matches_comfyui -- --ignored`
(with H3_REF_INPAINT=<out.safetensors>) compares the Rust port against them.
"""
import os, sys, types
sys.path.insert(0, "/mnt/extra/ai/ComfyUI")
os.chdir("/mnt/extra/ai/ComfyUI")
import torch
from safetensors.torch import save_file
import comfy.utils as utils
import comfy.model_base as mb
import comfy.ldm.minimax.model as mm

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/h3_inpaint_ref.safetensors"
torch.manual_seed(0)
F, H, W = 22, 96, 160                      # pixel mask
T, h, w, Ta = 7, 6, 10, 37                 # latent grid, audio frames
shift_v, shift_a = 12.0, 3.0
scale = shift_v / shift_a
sigma = 0.37

# a soft-edged blob moving over time
yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
mask = torch.stack([torch.clamp(1.5 - ((yy - 30 - t) ** 2 + (xx - 50 - 3 * t) ** 2).float().sqrt() / 12.0, 0, 1) for t in range(F)])

nv, na = torch.randn(1, 24, T, h, w), torch.randn(1, 32, 2, Ta)
Lv, La = torch.randn(1, 24, T, h, w), torch.randn(1, 32, 2, Ta)
xv, xa = torch.randn(1, 24, T, h, w), torch.randn(1, 32, 2, Ta)   # sampler state; audio carried
shapes = [list(nv.shape), list(na.shape)]

# the model object, without loading any weights
model = object.__new__(mb.MiniMaxH3)
torch.nn.Module.__init__(model)
model.latent_shapes = shapes
model.model_sampling = types.SimpleNamespace(shift=shift_v, audio_shift=shift_a, audio_scale=scale)
model.diffusion_model = types.SimpleNamespace(patch_size=(1, 2, 2))

# SetLatentNoiseMask + the sampler's per-stream preparation (audio generates everywhere)
nm = mask.reshape((-1, 1, H, W))
dm_v = utils.reshape_mask(nm, shapes[0])
dm_a = utils.reshape_mask(torch.ones(shapes[1]), shapes[1])
dm, _ = utils.pack_latents([dm_v, dm_a])
latent_image, _ = utils.pack_latents([Lv, La * scale])
noise, _ = utils.pack_latents([nv, na])
x, _ = utils.pack_latents([xv, xa])

tok = model._token_grid_masks(dm, shapes)
inj = model.scale_latent_inpaint(sigma=torch.tensor([sigma]), noise=noise, latent_image=latent_image, x=x, denoise_mask=dm)
x_in = x * dm + inj * (1.0 - dm)
x_in_v, x_in_a = utils.unpack_latents(x_in, shapes)
rows = mm.mask_row_values(tok[0][0, 0], T, h, w)

save_file({
    "mask": mask.contiguous(), "nv": nv, "na": na, "Lv": Lv, "La": La, "xv": xv, "xa": xa,
    "sigma": torch.tensor([sigma]), "dm_v": dm_v.contiguous(), "tok_v": tok[0].contiguous(),
    "x_in_v": x_in_v.contiguous(), "x_in_a": x_in_a.contiguous(),
    "rows": rows if rows is not None else torch.zeros(0),
}, out)
print("wrote", out, "token mask min", float(tok[0].min()), "rows", None if rows is None else tuple(rows.shape))
