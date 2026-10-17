import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsServer() {
  return (
    <DocsLayout>
      <H1>Web UI & API</H1>
      <Lead>
        <IC>h3 serve</IC> keeps every model loaded and renders requests one at a time. It serves a web studio at <IC>/</IC>, a JSON API
        under <IC>/v1</IC>, and its own OpenAPI document and reference page.
      </Lead>

      <Section>
        <H2>Start it</H2>
        <CopyBlock command={`h3 serve --addr 127.0.0.1:8188 --lora minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors --steps 8`} />
        <P>
          The text encoder and the VAE decoder are streamed from RAM and the DiT blocks fill the rest of the GPU, so one process holds
          everything. A warm 8-step 640×352 request takes about 12 s end to end on an RTX 4090. Open <IC>http://127.0.0.1:8188/</IC>
          for the studio and <IC>/docs</IC> for the API reference (rendered from <IC>/openapi.json</IC>).
        </P>
      </Section>

      <Section>
        <H2>The studio</H2>
        <P>
          Compose in one of three modes: <b class="font-medium text-ink">Text</b>, <b class="font-medium text-ink">Frames</b> (first and last
          keyframes) or <b class="font-medium text-ink">References</b> (pictures, videos and voices, each with a clickable tag for the prompt). The
          inspector sets format, resolution, duration, steps, seed, LoRA (listed from your <IC>loras</IC> folder) and, under Advanced, the sampler,
          sigma shifts and CRF. While a clip renders you see the stage, the denoising step and an ETA; finished takes stay in a strip for replay,
          download, or to reuse their prompt and seed. Settings persist in the browser.
        </P>
        <P>
          <IC>Ctrl+Enter</IC> generates. Stop waiting abandons the request in the browser; the GPU still finishes the render it started.
        </P>
      </Section>

      <Section>
        <H2>Generate from anywhere</H2>
        <CopyBlock
          command={`curl -o out.mp4 -H 'content-type: application/json' localhost:8188/v1/generate \\
  -d '{"prompt": "A red fox trots through snow", "seconds": 5, "seed": 3}'`}
        />
        <FlagTable
          rows={[
            { flag: "POST /v1/generate", meaning: "JSON (or the prompt as a plain body) in, video/mp4 out. Media fields are base64 and may be data: URLs." },
            { flag: "GET /v1/info", meaning: "Models, defaults, the LoRAs it can load and the queue length." },
            { flag: "GET /v1/status", meaning: "What the engine is doing now: state, stage, done and total. Cheap to poll." },
            { flag: "GET /openapi.json, /docs", meaning: "The API as OpenAPI 3.1, and a reference page for it (Scalar, loaded from a CDN)." },
            { flag: "GET /health", meaning: "ok" },
          ]}
        />
        <P>
          Request fields: <IC>prompt</IC> (required), <IC>width</IC>, <IC>height</IC>, <IC>megapixels</IC>, <IC>aspect</IC>, <IC>seconds</IC> or <IC>length</IC>, <IC>steps</IC>, <IC>seed</IC>,
          <IC>sampler</IC>, <IC>shift_video</IC>, <IC>shift_audio</IC>, <IC>image</IC>, <IC>last_image</IC>, <IC>fit</IC>, <IC>ref_images</IC>, <IC>ref_videos</IC>, <IC>ref_audios</IC>, <IC>ref_video_mute</IC>,
          <IC>ref_image_size</IC>, <IC>lora</IC> (<IC>"none"</IC> turns the server default off), <IC>lora_strength</IC> and <IC>crf</IC>.
        </P>
        <Pre caption="GET /v1/status" lang="json">{`{ "state": "running", "queued": 1, "stage": "sampling", "done": 6, "total": 8 }`}</Pre>
      </Section>

      <Section>
        <H2>Developing the UI</H2>
        <P>
          The studio is Solid and Tailwind in <IC>ui/</IC>; <IC>pnpm build</IC> writes <IC>ui/dist</IC>, which <IC>build.rs</IC> embeds in the binary.
          <IC>pnpm dev</IC> serves it with hot reload and proxies the API to a running <IC>h3 serve</IC> (<IC>H3_URL</IC>).
        </P>
        <CopyBlock command={`cd ui && pnpm install && pnpm build   # then: cargo build --release`} />
      </Section>
    </DocsLayout>
  );
}
