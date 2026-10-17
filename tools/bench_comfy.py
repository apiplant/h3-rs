"""Time one DiT step in ComfyUI at given shapes. usage: bench_comfy.py"""
import os, sys, time
sys.path.insert(0, "/mnt/extra/ai/ComfyUI"); os.chdir("/mnt/extra/ai/ComfyUI")
sys.argv = sys.argv[:1] + ["--disable-all-custom-nodes"]
import torch
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd, comfy.model_management as mm
torch.set_grad_enabled(False); torch.inference_mode().__enter__()
model = comfy.sd.load_diffusion_model("/mnt/extra/ai/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
mm.load_models_gpu([model], memory_required=7e9); dm = model.model.diffusion_model; dev = mm.get_torch_device()
print("loaded, VRAM", torch.cuda.memory_allocated() / 1e9)
for (T, H, W, TA, name) in [(37, 48, 84, 207, "1344x768x124f")]:
    ctx = torch.randn(1, 30, 5120, dtype=torch.bfloat16, device=dev)
    text = dm.preprocess_text_embeds(ctx)
    v = torch.randn(1, 24, T, H, W, device=dev); a = torch.randn(1, 32, 2, TA, device=dev)
    ts = []
    for i in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        dm.forward([v, a], torch.tensor([700.0], device=dev), text, transformer_options={}, minimax_payload={"audio_scale": 4.0, "seed": 0})
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    print(name, "s/step", ["%.2f" % t for t in ts], "peak VRAM %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
