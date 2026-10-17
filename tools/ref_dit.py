"""Reference dumps from ComfyUI for h3-rs parity tests.

usage: python tools/ref_dit.py <out_dir>
Writes:
  dequant.safetensors : dequantized weights (int8 convrot DiT qkv, nvfp4 TE down_proj)
  dit_fwd.safetensors : inputs + outputs of one MiniMaxH3Model.forward at a tiny shape
"""
import os, sys
sys.path.insert(0, "/mnt/extra/ai/ComfyUI")
os.chdir("/mnt/extra/ai/ComfyUI")
sys.argv = [sys.argv[0]] + ["--disable-all-custom-nodes"] + sys.argv[2:] if len(sys.argv) > 2 else sys.argv[:1] + ["--disable-all-custom-nodes"]
import torch
from safetensors.torch import save_file, safe_open
import comfy.options
comfy.options.enable_args_parsing()
import comfy.sd, comfy.utils, comfy.model_management as mm
import comfy.quant_ops as qo

out = os.environ.get("H3_REF", "/tmp/h3ref")
os.makedirs(out, exist_ok=True)
M = "/mnt/extra/ai/ComfyUI/models"
DIT = f"{M}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE = f"{M}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

what = os.environ.get("H3_WHAT", "dequant,dit").split(",")

if "dequant" in what:
    d = {}
    with safe_open(DIT, "pt") as f:
        q = f.get_tensor("blocks.0.attn.qkv_proj.weight").cuda()
        s = f.get_tensor("blocks.0.attn.qkv_proj.weight_scale").cuda()
        d["dit_qkv_plain"] = qo.ck.dequantize_int8_convrot_weight(q, s, 256).float().cpu() if hasattr(qo.ck, "dequantize_int8_convrot_weight") else torch.ops.comfy_kitchen.dequantize_int8_convrot_weight(q, s, 256).float().cpu()
    with safe_open(TE, "pt") as f:
        p = "model.layers.0.mlp.down_proj"
        q = f.get_tensor(p + ".weight").cuda()
        bs = f.get_tensor(p + ".weight_scale").cuda().view(torch.float8_e4m3fn)
        g = f.get_tensor(p + ".weight_scale_2").cuda()
        d["te_down_plain"] = qo.ck.dequantize_nvfp4(q, g, bs, torch.bfloat16).float().cpu()
    save_file(d, f"{out}/dequant.safetensors")
    print("dequant done", {k: tuple(v.shape) for k, v in d.items()})

if "dit" in what:
    model = comfy.sd.load_diffusion_model(DIT)
    mm.load_models_gpu([model])
    dm = model.model.diffusion_model
    dev = mm.get_torch_device()
    if os.environ.get("H3_FULLPREC"):
        n = 0
        for mod in dm.modules():
            if getattr(mod, "layout_type", None) is not None:
                mod._full_precision_mm = True; n += 1
        print("full precision matmul on", n, "layers")
    captured = {}
    for i in (0, 1, 49):
        dm.blocks[i].register_forward_hook(lambda m, inp, o, i=i: captured.__setitem__(f"block{i}", o.float().cpu().clone()))
    g = torch.Generator("cpu").manual_seed(0)
    L, T, H, W, TA = 40, 2, 8, 12, 8
    ctx = torch.randn(1, L, 5120, generator=g).to(torch.bfloat16)
    v = torch.randn(1, 24, T, H, W, generator=g)
    a = torch.randn(1, 32, 2, TA, generator=g)
    sigma = 0.7
    with torch.no_grad():
        text = dm.preprocess_text_embeds(ctx.to(dev))
        payload = {"audio_scale": 4.0, "seed": 0}
        o = dm.forward([v.to(dev), a.to(dev)], torch.tensor([sigma * 1000.0], device=dev), text,
                       transformer_options={}, minimax_payload=payload)
    save_file({"ctx": ctx.float(), "text_refined": text[0].float().cpu(), "v": v, "a": a,
               "out_v": o[0].float().cpu().contiguous(), "out_a": o[1].float().cpu().contiguous(),
               "sigma": torch.tensor([sigma]), **captured}, f"{out}/dit_fwd.safetensors")
    print("dit done", o[0].shape, o[1].shape, o[0].abs().mean().item(), o[1].abs().mean().item())
