import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>h3-rs</IC> crate (library name <IC>h3_rs</IC>) is the engine behind the <IC>h3</IC> command: model loading, the
        quantized DiT, the text encoder, both VAEs, the sampler and the MP4 writer.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <P>
          The HTTP server (ntex, <IC>h3_rs::serve</IC>) and the <IC>h3</IC> binary are behind the default <IC>serve</IC> feature;
          <IC>flash-attn</IC> is a second default feature (it adds a long build, and the engine's own int8 attention does not need
          it). The crate needs the CUDA 13 toolkit and, at run time, <IC>ffmpeg</IC>. The tokenizer is embedded.
        </P>
        <CopyBlock command={`cargo add h3-rs --no-default-features`} />
      </Section>

      <Section>
        <H2>Render a clip</H2>
        <Pre caption="src/main.rs" lang="rust">{`use h3_rs::config::Preset;
use h3_rs::pipeline::{Engine, ModelPaths, Request};

let paths = ModelPaths::resolve(
    "/models".as_ref(),
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
)?;
let dev = candle_core::Device::new_cuda(0)?;
h3_rs::kernels::keep_pool_memory(&dev);

// keep = true: models stay loaded between calls
let mut engine = Engine::new(paths, None, Preset::Gb24.config(), dev, true);

let req = Request {
    prompt: "A red fox trots through fresh snow at dawn, crunching footsteps.".into(),
    seed: 3,
    steps: 20,
    output: "fox.mp4".into(),
    ..Default::default()
};
engine.generate(&req)?;`}</Pre>
        <P>
          <IC>Request</IC> mirrors the command line: keyframes (<IC>first_frame</IC>, <IC>last_frame</IC>), references (<IC>refs</IC>, with
          <IC>RefInput::Image</IC>, <IC>Video</IC> and <IC>Audio</IC>), <IC>lora</IC> as <IC>(path, strength)</IC>, <IC>length</IC> in frames at 24 fps, and so on. Pass the
          reference-to-video checkpoint as the second argument of <IC>Engine::new</IC> to enable references.
        </P>
        <P>
          While a request runs, <IC>h3_rs::progress::snapshot()</IC> reports the stage and step, which is what the server's <IC>/v1/status</IC>
          publishes.
        </P>
      </Section>
    </DocsLayout>
  );
}
