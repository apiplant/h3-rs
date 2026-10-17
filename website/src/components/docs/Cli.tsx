import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>h3</H1>
      <Lead>Render a clip from the command line, serve the engine over HTTP, or decode saved latents.</Lead>

      <Section>
        <H2>Examples</H2>
        <CopyBlock
          command={`# text -> video+audio (defaults: 1344x768, 124 frames = 5.2 s, 20 steps, res_multistep)
h3 generate -p "A red fox trots through fresh snow at dawn, crunching footsteps." -o fox.mp4

# first (and/or last) keyframe: the canvas follows the image's aspect ratio
h3 generate -p "..." --image first.png --last-image last.png

# references: images, then videos, then audio files, numbered per type
h3 generate -p "<Picture 1> walks through the scene of <Video 1>, speaking with the voice of <Audio 2>" \\
  --ref-image person.png --ref-video scene.mp4 --ref-audio voice.wav

# official turbo LoRAs: 8 steps for text/frames, 4 for references
h3 generate -p "..." --lora minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors --steps 8

# save the latents, then decode again later
h3 generate -p "..." --save-latents run.safetensors
h3 decode run.safetensors -o run.mp4`}
        />
        <P>
          A video's soundtrack is the <IC>{"<Audio j>"}</IC> just before its <IC>{"<Video k>"}</IC>; <IC>--ref-video-mute</IC> ignores it.
        </P>
      </Section>

      <Section>
        <H2>generate</H2>
        <FlagTable
          rows={[
            { flag: "-p, --prompt TEXT", meaning: "What to generate (required). Describe the sound too." },
            { flag: "-o, --output FILE", meaning: "Output MP4 (default h3_output.mp4); a .wav of the soundtrack is written beside it." },
            { flag: "--seconds S, --length N", meaning: "Duration, or a frame count at 24 fps. Snapped up to the 17k+5 grid; 124 frames is about 5 s." },
            { flag: "--width, --height", meaning: "Canvas in pixels, snapped to 32. Default: from the keyframe or aspect, else 1344×768." },
            { flag: "--aspect W:H, --megapixels M", meaning: "Canvas for an aspect ratio at M × 1024² pixels (default 1)." },
            { flag: "--image, --last-image", meaning: "First and last keyframes. --fit stretch|crop chooses how they fill the canvas." },
            { flag: "--ref-image, --ref-video, --ref-audio", meaning: "References (up to 9 pictures, 3 videos, 3 voices). --ref-image-size match|max, --ref-video-mute." },
            { flag: "--audio-guide FILE", meaning: "Audio anchored at frame 0 (speech for a talking image). Use with --image and set --seconds to its length." },
            { flag: "--steps 20, --seed 1", meaning: "Sampling steps and noise seed (the same seed as ComfyUI gives the same starting latent)." },
            { flag: "--sampler res-multistep|euler", meaning: "Sampler. --shift-video 12 and --shift-audio 3 set the sigma shifts." },
            { flag: "--lora FILE, --lora-strength 1", meaning: "A LoRA, by path or by name in models/loras. Merged into the weights at load." },
            { flag: "--crf 18", meaning: "H.264 quality." },
            { flag: "--save-latents FILE", meaning: "Also save the sampled latents for h3 decode." },
          ]}
        />
      </Section>

      <Section>
        <H2>Models and memory</H2>
        <FlagTable
          rows={[
            { flag: "--models-dir DIR", meaning: "The ComfyUI models folder. Default: $H3_MODELS_DIR, ./models or ~/ComfyUI/models when they exist, else ~/.cache/h3-rs/models." },
            { flag: "--dit, --text-encoder, --video-vae, --audio-vae", meaning: "Override single files by name or path." },
            { flag: "--preset gb24|gb16|gb12|gb8|minimal|auto", meaning: "Memory preset (default gb24). See Memory presets." },
            { flag: "--config FILE.toml", meaning: "A [memory] table; overrides the preset." },
            { flag: "--dit-gpu-blocks, --te-gpu-layers, --offload ram|disk, --mlp-chunk", meaning: "Override single memory knobs." },
            { flag: "--device N", meaning: "CUDA device." },
          ]}
        />
      </Section>

      <Section>
        <H2>serve, decode, presets</H2>
        <FlagTable
          rows={[
            { flag: "h3 serve --addr 127.0.0.1:8188", meaning: "The HTTP server and web UI; see Web UI & API. --steps, --lora and --lora-strength set the defaults for requests that leave them out." },
            { flag: "h3 decode LATENTS -o OUT.mp4", meaning: "Decode latents saved by --save-latents." },
            { flag: "h3 presets", meaning: "Print the memory presets as TOML." },
          ]}
        />
        <CopyBlock command={`h3 --version`} />
      </Section>
    </DocsLayout>
  );
}
