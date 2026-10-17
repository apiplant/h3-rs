/* The HTTP API of `h3 serve`; /openapi.json describes the same endpoints. */

export interface GenerateRequest {
  prompt: string;
  width?: number;
  height?: number;
  megapixels?: number;
  aspect?: string;
  seconds?: number;
  steps?: number;
  seed?: number;
  sampler?: "res_multistep" | "euler";
  shift_video?: number;
  shift_audio?: number;
  image?: string;
  last_image?: string;
  fit?: "stretch" | "crop";
  ref_images?: string[];
  ref_videos?: string[];
  ref_audios?: string[];
  ref_video_mute?: boolean;
  ref_image_size?: "match" | "max";
  lora?: string;
  lora_strength?: number;
  crf?: number;
}

export interface Info {
  version: string;
  dit: string;
  ref_dit: string | null;
  text_encoder: string;
  default_steps: number;
  default_lora: string | null;
  loras: string[];
  queued: number;
}

export interface Status {
  state: "idle" | "starting" | "running";
  queued: number;
  stage: "idle" | "encoding" | "sampling" | "decoding" | "muxing";
  done: number;
  total: number;
}

async function check(r: Response): Promise<Response> {
  if (!r.ok) throw new Error((await r.text()).trim() || `${r.status} ${r.statusText}`);
  return r;
}

export async function info(): Promise<Info> {
  return (await check(await fetch("/v1/info"))).json();
}

export async function status(): Promise<Status> {
  return (await check(await fetch("/v1/status", { cache: "no-store" }))).json();
}

/** POST /v1/generate; resolves when the video is finished. */
export async function generate(req: GenerateRequest, signal?: AbortSignal): Promise<Blob> {
  const r = await check(
    await fetch("/v1/generate", { method: "POST", body: JSON.stringify(req), headers: { "content-type": "application/json" }, signal }),
  );
  return r.blob();
}
