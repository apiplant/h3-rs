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
  /** Images / clips ("image") and soundtracks ("audio") anchored at a frame. */
  guides?: { kind: "image" | "audio"; data: string; frame: number }[];
  source?: string;
  source_mute?: boolean;
  mask?: string;
  mask_audio?: "follow" | "keep" | "generate";
  controlnet?: string;
  control_video?: string;
  control_strength?: number;
  control_start?: number;
  control_end?: number;
  lora?: string;
  lora_strength?: number;
  /** Sample at 1/N of the size, upscale the latent, refine at full size. */
  upscale?: number;
  /** With `upscale`: fraction of the steps sampled at the low resolution. */
  upscale_split?: number;
  crf?: number;
}

export interface Info {
  version: string;
  dit: string;
  ref_dit: string | null;
  text_encoder: string;
  default_steps: number;
  default_lora: string | null;
  /** Latent upscaler checkpoint, when the server found one. */
  upscaler: string | null;
  loras: string[];
  /** Fun ControlNet checkpoints in models/model_patches. */
  controlnets?: string[];
  queued: number;
}

export interface Status {
  state: "idle" | "starting" | "running";
  queued: number;
  stage: "idle" | "encoding" | "sampling" | "upscaling" | "decoding" | "muxing" | "downloading";
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

export async function cancel(): Promise<void> {
  await fetch("/v1/cancel", { method: "POST" }).catch(() => {});
}

export async function status(): Promise<Status> {
  return (await check(await fetch("/v1/status", { cache: "no-store" }))).json();
}

/** DELETE /v1/jobs: the server keeps a finished render's folder until the user deletes the take. */
export async function deleteJob(job: string): Promise<void> {
  await check(await fetch(`/v1/jobs?id=${encodeURIComponent(job)}`, { method: "DELETE" }));
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((ok, no) => {
    const t = setTimeout(ok, ms);
    signal?.addEventListener("abort", () => (clearTimeout(t), no(new DOMException("aborted", "AbortError"))), { once: true });
  });

/**
 * Queue the request (POST /v1/jobs), poll the job until it is finished, then fetch the video. No single HTTP request
 * stays open for the whole render, so a dropped connection while it runs costs nothing (`job`: its folder on the server).
 */
export async function generate(req: GenerateRequest, signal?: AbortSignal): Promise<{ blob: Blob; job: string }> {
  const { id } = await (await check(await fetch("/v1/jobs", { method: "POST", body: JSON.stringify(req), headers: { "content-type": "application/json" }, signal }))).json();
  for (;;) {
    await sleep(1000, signal);
    let s: { state: string; error?: string };
    try {
      s = await (await check(await fetch(`/v1/jobs?id=${encodeURIComponent(id)}`, { cache: "no-store", signal }))).json();
    } catch (e: any) {
      if (e.name === "AbortError") throw e;
      continue; // server busy or briefly unreachable: keep waiting
    }
    if (s.state === "failed") throw new Error(s.error || "render failed");
    if (s.state === "done") break;
  }
  for (let tries = 0; ; tries++) {
    try {
      const r = await check(await fetch(`/v1/jobs/result?id=${encodeURIComponent(id)}`, { signal }));
      return { blob: await r.blob(), job: id };
    } catch (e: any) {
      if (e.name === "AbortError" || tries >= 5) throw e;
      await sleep(1000, signal);
    }
  }
}
