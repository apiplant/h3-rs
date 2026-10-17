"""Build assets/tokenizer.json from ComfyUI's qwen25_tokenizer (+ MiniMax H3 extra tokens) and verify ids."""
import sys, json
sys.path.insert(0, "/mnt/extra/ai/ComfyUI")
from transformers import Qwen2TokenizerFast, Qwen2Tokenizer
d = "/mnt/extra/ai/ComfyUI/comfy/text_encoders/qwen25_tokenizer"
extra = ["<d>", "</d>", "<|cutoff|>", "<|lyrics_start|>", "<|lyrics_end|>", "<|caption_start|>", "<|caption_end|>"]
fast = Qwen2TokenizerFast.from_pretrained(d)
slow = Qwen2Tokenizer.from_pretrained(d)
for t in (fast, slow):
    t.add_special_tokens({"additional_special_tokens": extra})
fast.save_pretrained("/mnt/extra/projects/h3-rs/assets/tok_tmp")
import shutil; shutil.copy("/mnt/extra/projects/h3-rs/assets/tok_tmp/tokenizer.json", "/mnt/extra/projects/h3-rs/assets/tokenizer.json"); shutil.rmtree("/mnt/extra/projects/h3-rs/assets/tok_tmp")
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/mnt/extra/projects/h3-rs/assets/tokenizer.json")
import comfy.text_encoders.minimax as mm
ct = mm.MiniMaxH3Tokenizer()
tests = ["<Picture 1>: ", "A cat 🐈 walks, 'quickly' — über café!\n\nTimeline:\n[0s-1s] VHS static opens   the frame.", "<0.5 seconds>", "<d>hi</d> <|caption_start|>x", "  leading spaces and 1234567 numbers"]
ok = True
for s in tests:
    a = tk.encode(s, add_special_tokens=False).ids
    b = [t[0] for t in ct.qwen3vl_32b.tokenize_with_weights(s, disable_weights=True)[0]]
    print(a == b, len(a), s[:40].encode()); ok &= a == b
    if a != b: print(a, b)
print("vision ids", tk.token_to_id("<|vision_start|>"), tk.token_to_id("<|vision_end|>"), tk.token_to_id("<d>"))
print("ALL OK" if ok else "MISMATCH")
