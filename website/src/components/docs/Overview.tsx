import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        h3-rs runs MiniMax's H3 video model, with its stereo soundtrack, on an NVIDIA GPU in Rust (candle plus hand-written
        CUDA). It is a port of ComfyUI's implementation and loads the ComfyUI model files as they are. It ships as a library
        crate and the <IC>h3</IC> command.
      </Lead>

      <Section>
        <H2>The model files</H2>
        <P>Put them in a ComfyUI-style <IC>models</IC> folder and point <IC>H3_MODELS_DIR</IC> (or <IC>--models-dir</IC>) at it.</P>
        <FlagTable
          rows={[
            { flag: "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", meaning: "The DiT for text and keyframes: int8 per row with a ConvRot (Hadamard-256) rotation." },
            { flag: "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", meaning: "The DiT for reference pictures, videos and voices. Picked automatically when references are given." },
            { flag: "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", meaning: "Qwen3-VL 32B in NVFP4 with AWQ and int8 embeddings." },
            { flag: "vae/minimax_h3_video_vae_fp16.safetensors", meaning: "The video VAE (an int8_convrot variant also loads)." },
            { flag: "vae/minimax_h3_audio_vae_fp32.safetensors", meaning: "The audio VAE (BigVGAN decoder, stereo)." },
            { flag: "loras/*.safetensors", meaning: "Optional ComfyUI LoRAs, including head-bank reshapes. The official turbo LoRAs are on Hugging Face under Comfy-Org/MiniMax-H3." },
          ]}
        />
        <P>
          Quantized weights stay quantized in memory and are dequantized per layer by NVRTC kernels. Frame counts snap up to the
          model's 17k+5 grid at 24 fps and sizes to multiples of 32. Noise matches ComfyUI's <IC>RandomNoise</IC>, so a seed gives
          the same starting latent as in ComfyUI.
        </P>
      </Section>

      <Section>
        <H2>Speed (RTX 4090, one DiT step)</H2>
        <FlagTable
          rows={[
            { flag: "1344×768, 124 frames", meaning: "13.0 s in h3-rs, 15.6 s in ComfyUI" },
            { flag: "640×352, 56 frames", meaning: "0.52 s in h3-rs, 0.47 s in ComfyUI" },
          ]}
        />
        <UL>
          <LI>int8 weights run on int8 tensor cores: a fused GEMM quantizes activations per row and writes bf16 directly, with the ConvRot rotation done warp-level inside the quantizer and SwiGLU fused into fc1's epilogue.</LI>
          <LI>DiT attention is an int8 Q·Kᵀ / fp16 P·V flash-style kernel, about 1.25× flash-attn 2 at full size.</LI>
          <LI>LoRAs are merged into the int8 weights at load (dequantize, add, requantize), so they cost nothing per step.</LI>
          <LI>Fused RMSNorm and adaLN modulate; streamed blocks upload on a second CUDA stream from pinned RAM.</LI>
        </UL>
      </Section>

      <Section>
        <H2>Requirements</H2>
        <UL>
          <LI>An NVIDIA GPU, Ampere or newer. 24 GB is comfortable; <IC>gb8</IC> renders full size in a 6.5 GB peak by streaming weights from disk (see <a class="text-accent hover:text-accent-dim" href="/docs/memory">memory presets</a>).</LI>
          <LI>Linux x86_64 with a driver for CUDA 13, and <IC>ffmpeg</IC> on PATH (it reads reference media and writes the MP4).</LI>
          <LI>16 to 48 GB of RAM depending on the preset.</LI>
        </UL>
        <CopyBlock command={`H3_MODELS_DIR=~/ComfyUI/models h3 generate -p "A red fox trots through fresh snow at dawn." -o fox.mp4`} />
      </Section>

      <Section>
        <H2>Not yet ported</H2>
        <P>
          <IC>MiniMaxH3AddGuide</IC> clips and audio at arbitrary frames, Fun ControlNet and denoise masks. The older{" "}
          <IC>MiniMax-H3-FL2VA-Acc-8Step_comfy</IC> LoRA targets adaLN at a width the pruned checkpoint doesn't have, so those
          targets are skipped (as in ComfyUI); prefer the official turbo LoRAs.
        </P>
      </Section>
    </DocsLayout>
  );
}
