import { For, Show, createEffect, createMemo, createSignal, onSettled, untrack } from "solid-js";
import type { JSX } from "@solidjs/web";
import { cancel as cancelRender, deleteJob, generate, info as fetchInfo, status as fetchStatus, type GenerateRequest, type Info, type Status } from "./api";
import { FPS, canvas, dataUrl, fmtTime, frames, inspect, type Picked } from "./media";
import {
  btn, btnGhost, btnIcon, btnLg, btnPrimary, btnSm, btnSmIcon, btnStop, chipBtn, dot, editorBack, editorInput, errorBox, input, kbd,
  panel, panelHead, panelTitle, sectionLabel, select, take, tokAud, tokPic, tokVid,
} from "./styles";
import { Icon, Logo, Ratio, Segmented, Slider, Toggle } from "./ui";

type Mode = "text" | "frames" | "refs" | "edit";

/** A MiniMaxH3AddGuide: a picture, clip or soundtrack anchored at a frame. */
interface Guide {
  p: Picked;
  kind: "image" | "video" | "audio";
  frame: number;
}

interface Take {
  id: number;
  /** The render's folder on the server (deleted with the take). */
  job: string;
  url: string;
  prompt: string;
  seed: number;
  spec: string;
  seconds: number;
  took: number;
  mode: Mode;
}

const EXAMPLES = [
  "A red fox trots through fresh snow at dawn, breath steaming. Soft wind and crunching footsteps.",
  "A street violinist plays under a neon-lit awning in heavy rain, cars hissing past, the melody echoing off wet brick.",
  "Waves crash against black rocks at sunset; gulls cry overhead and the surf roars, handheld camera drifting forward.",
  "A barista pours latte art in a sunlit cafe. The espresso machine hisses, cups clink, a woman laughs softly nearby.",
];
const ASPECTS: [string, number, number][] = [
  ["16:9", 16, 9], ["9:16", 9, 16], ["1:1", 1, 1], ["4:3", 4, 3], ["3:4", 3, 4], ["21:9", 21, 9],
];
const QUALITY = [
  { value: 0.25, label: "Draft", hint: "0.25 MP: the fastest way to try a prompt" },
  { value: 0.5, label: "Balanced", hint: "0.5 MP" },
  { value: 1, label: "HD", hint: "1 MP: the model's native size" },
];
const STAGES = [
  { id: "encoding", label: "Reading the prompt" },
  { id: "sampling", label: "Denoising" },
  { id: "decoding", label: "Decoding video" },
  { id: "muxing", label: "Mixing audio" },
] as const;
const STORE = "h3.ui.v1";
let nextId = 0;

interface Saved {
  mode: Mode;
  prompt: string;
  aspect: string;
  mp: number;
  seconds: number;
  steps: string;
  seed: string;
  lora: string;
  loraStrength: number;
  sampler: "res_multistep" | "euler";
  shiftV: number;
  shiftA: number;
  crf: number;
  fit: "" | "stretch" | "crop";
  refSize: "match" | "max";
  mute: boolean;
  custom: boolean;
  cw: number;
  ch: number;
  upscale: boolean;
  factor: number;
  split: number;
}

function loadSaved(): Partial<Saved> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "{}");
  } catch {
    return {};
  }
}

const TAG_RE = /<(Picture|Video|Audio) \d+>/g;

/** The prompt as plain text and highlighted reference tags, for the layer behind the editor. */
function tokens(t: string): { t: string; kind?: "Picture" | "Video" | "Audio" }[] {
  const out: { t: string; kind?: "Picture" | "Video" | "Audio" }[] = [];
  let at = 0;
  for (const m of t.matchAll(TAG_RE)) {
    if (m.index > at) out.push({ t: t.slice(at, m.index) });
    out.push({ t: m[0], kind: m[1] as "Picture" | "Video" | "Audio" });
    at = m.index + m[0].length;
  }
  out.push({ t: t.slice(at) + "\n" });
  return out;
}

/** A click-or-drop target that hands back the files it received. */
function Drop(props: { accept: string; multiple?: boolean; onFiles: (f: File[]) => void; class?: string; children: JSX.Element; title?: string }) {
  let file!: HTMLInputElement;
  const [over, setOver] = createSignal(false);
  return (
    <div
      role="button"
      tabindex="0"
      title={props.title}
      class={`${props.class ?? ""} cursor-pointer transition-colors ${over() ? "!border-accent !bg-accent/10" : ""}`}
      onClick={() => file.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), file.click())}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        props.onFiles([...(e.dataTransfer?.files ?? [])]);
      }}
    >
      <input
        ref={file}
        type="file"
        class="hidden"
        accept={props.accept}
        multiple={props.multiple}
        onChange={(e) => {
          props.onFiles([...(e.currentTarget.files ?? [])]);
          e.currentTarget.value = "";
        }}
      />
      {props.children}
    </div>
  );
}

/** The attached file shown in the preview overlay, if any. */
const [preview, setPreview] = createSignal<{ p: Picked; kind: "image" | "video" | "audio" } | null>(null);

/** Full-size view of an attached image, a player for an attached video or audio file. Click outside or Esc closes it. */
function Preview() {
  return (
    <Show when={preview()}>
      {(v) => (
        <div
          class="fixed inset-0 z-50 grid place-items-center bg-ink-950/90 p-6 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label={`Preview of ${v().p.name}`}
          onClick={() => setPreview(null)}
        >
          <button class={`${btn} ${btnIcon} absolute top-4 right-4`} aria-label="Close preview" onClick={() => setPreview(null)}>
            <Icon name="x" size={16} />
          </button>
          <div class="flex max-h-full max-w-full flex-col items-center gap-3" onClick={(e) => e.stopPropagation()}>
            <Show when={v().kind === "image"}>
              <img src={v().p.url} alt={v().p.name} class="max-h-[85vh] max-w-full rounded-lg object-contain" />
            </Show>
            <Show when={v().kind === "video"}>
              <video src={v().p.url} controls autoplay playsinline class="max-h-[85vh] max-w-full rounded-lg" />
            </Show>
            <Show when={v().kind === "audio"}>
              <div class="flex flex-col items-center gap-4 rounded-xl border border-ink-700 bg-ink-850 p-8">
                <span class="text-aud">
                  <Icon name="music" size={40} />
                </span>
                <audio src={v().p.url} controls autoplay class="w-80 max-w-full" />
              </div>
            </Show>
            <p class="font-mono text-xs text-ink-400">
              {v().p.name}
              {v().p.w && v().p.h ? ` · ${v().p.w}×${v().p.h}` : ""}
              {v().p.seconds && v().kind !== "image" ? ` · ${fmtTime(v().p.seconds!)}` : ""}
            </p>
          </div>
        </div>
      )}
    </Show>
  );
}

function Thumb(props: { p: Picked; kind: "image" | "video" | "audio"; tag?: string; onTag?: () => void; onRemove: () => void; class?: string }) {
  return (
    <div
      role="button"
      tabindex="0"
      title={`Preview ${props.p.name}`}
      class={`group relative cursor-zoom-in overflow-hidden rounded-lg border border-ink-700 bg-ink-850 ${props.class ?? ""}`}
      onClick={(e) => {
        e.stopPropagation();
        setPreview({ p: props.p, kind: props.kind });
      }}
      onKeyDown={(e) => e.target === e.currentTarget && (e.key === "Enter" || e.key === " ") && (e.preventDefault(), setPreview({ p: props.p, kind: props.kind }))}
    >
      <Show when={props.kind === "image"}>
        <img src={props.p.url} alt={props.p.name} class="size-full object-cover" />
      </Show>
      <Show when={props.kind === "video"}>
        <video src={props.p.url} muted preload="metadata" class="size-full object-cover" />
      </Show>
      <Show when={props.kind === "audio"}>
        <div class="grid size-full place-items-center text-aud">
          <Icon name="music" size={22} />
        </div>
      </Show>
      <button
        class="absolute top-1 right-1 grid size-5 cursor-pointer place-items-center rounded-full bg-ink-950/80 text-ink-200 opacity-0 transition-opacity hover:text-white group-hover:opacity-100 focus-visible:opacity-100"
        aria-label={`Remove ${props.p.name}`}
        onClick={(e) => {
          e.stopPropagation();
          props.onRemove();
        }}
      >
        <Icon name="x" size={11} />
      </button>
      <Show when={props.tag}>
        <Show
          when={props.onTag}
          fallback={<span class="absolute inset-x-1 bottom-1 truncate rounded bg-ink-950/85 px-1 py-px font-mono text-[10px] text-ink-200">{props.tag}</span>}
        >
          <button
            class="absolute inset-x-1 bottom-1 cursor-pointer truncate rounded bg-ink-950/85 px-1 py-px text-left font-mono text-[10px] text-ink-100 hover:bg-accent hover:text-ink-950"
            title="Insert this tag into the prompt"
            onClick={(e) => {
              e.stopPropagation();
              props.onTag!();
            }}
          >
            {props.tag}
          </button>
        </Show>
      </Show>
      <Show when={props.p.seconds && props.kind !== "image"}>
        <span class="absolute top-1 left-1 rounded bg-ink-950/80 px-1 font-mono text-[10px] text-ink-300">{fmtTime(props.p.seconds!)}</span>
      </Show>
    </div>
  );
}

export function App() {
  const saved = loadSaved();
  const [info, setInfo] = createSignal<Info | null>(null);
  const [online, setOnline] = createSignal(true);
  const [mode, setMode] = createSignal<Mode>(saved.mode ?? "text");
  const [prompt, setPrompt] = createSignal(saved.prompt ?? EXAMPLES[0]);
  const [aspect, setAspect] = createSignal(saved.aspect ?? "16:9");
  const [mp, setMp] = createSignal(saved.mp ?? 0.5);
  const [seconds, setSeconds] = createSignal(saved.seconds ?? 5);
  const [steps, setSteps] = createSignal(saved.steps ?? "");
  const [seed, setSeed] = createSignal(saved.seed ?? "");
  const [lora, setLora] = createSignal(saved.lora ?? "");
  const [loraStrength, setLoraStrength] = createSignal(saved.loraStrength ?? 1);
  const [sampler, setSampler] = createSignal<Saved["sampler"]>(saved.sampler ?? "res_multistep");
  const [shiftV, setShiftV] = createSignal(saved.shiftV ?? 12);
  const [shiftA, setShiftA] = createSignal(saved.shiftA ?? 3);
  const [crf, setCrf] = createSignal(saved.crf ?? 18);
  const [fit, setFit] = createSignal<Saved["fit"]>(saved.fit ?? "");
  const [refSize, setRefSize] = createSignal<Saved["refSize"]>(saved.refSize ?? "match");
  const [mute, setMute] = createSignal(saved.mute ?? false);
  const [custom, setCustom] = createSignal(saved.custom ?? false);
  const [cw, setCw] = createSignal(saved.cw ?? 1280);
  const [ch, setCh] = createSignal(saved.ch ?? 704);
  const [upscale, setUpscale] = createSignal(saved.upscale ?? false);
  const [factor, setFactor] = createSignal(saved.factor ?? 1.5);
  const [split, setSplit] = createSignal(saved.split ?? 0.5);
  const [advanced, setAdvanced] = createSignal(false);

  const [first, setFirst] = createSignal<Picked | null>(null);
  const [last, setLast] = createSignal<Picked | null>(null);
  const [refImgs, setRefImgs] = createSignal<Picked[]>([]);
  const [refVids, setRefVids] = createSignal<Picked[]>([]);
  const [refAuds, setRefAuds] = createSignal<Picked[]>([]);
  const [guides, setGuides] = createSignal<Guide[]>([]);
  const [src, setSrc] = createSignal<Picked | null>(null);
  const [maskF, setMaskF] = createSignal<Picked | null>(null);
  const [srcMute, setSrcMute] = createSignal(false);
  const [maskAudio, setMaskAudio] = createSignal<"follow" | "keep" | "generate">("follow");
  const [cnet, setCnet] = createSignal("");
  const [ctlVideo, setCtlVideo] = createSignal<Picked | null>(null);
  const [ctlStrength, setCtlStrength] = createSignal(1);
  const [ctlRange, setCtlRange] = createSignal<[number, number]>([0, 1]);

  const [busy, setBusy] = createSignal(false);
  const [elapsed, setElapsed] = createSignal(0);
  const [st, setSt] = createSignal<Status | null>(null);
  const [rate, setRate] = createSignal(0);
  const [error, setError] = createSignal("");
  const [takes, setTakes] = createSignal<Take[]>([]);
  const [current, setCurrent] = createSignal<number | null>(null);

  let abort: AbortController | null = null;
  let ta!: HTMLTextAreaElement;
  let back!: HTMLDivElement;
  let touched = false;

  const segments = createMemo(() => tokens(prompt()));
  const shown = createMemo(() => takes().find((t) => t.id === current()) ?? null);

  // ---- what the next render will be, as far as the browser can tell ----
  const ratio = createMemo<[number, number]>(() => {
    const f = mode() === "frames" ? (first() ?? last()) : null;
    if (f?.w && f?.h) return [f.w, f.h];
    const a = ASPECTS.find((x) => x[0] === aspect()) ?? ASPECTS[0];
    return [a[1], a[2]];
  });
  // a manual size snaps to the model's 32 px grid, like the engine does
  const snap = (v: number) => Math.max(32, Math.round((Number.isFinite(v) ? v : 0) / 32) * 32);
  const size = createMemo<[number, number]>(() => (custom() ? [snap(cw()), snap(ch())] : canvas(ratio()[0], ratio()[1], mp())));
  const canUpscale = () => !!info()?.upscaler;
  const upscaling = () => upscale() && canUpscale();
  /** The delivered canvas: the selected one, times the factor when the upscaler is on (the engine rounds to 32). */
  const outSize = createMemo<[number, number]>(() =>
    upscaling() ? [Math.round((size()[0] * factor()) / 32) * 32, Math.round((size()[1] * factor()) / 32) * 32] : size(),
  );
  const nFrames = createMemo(() => frames(seconds()));
  const spec = createMemo(() => `${outSize()[0]}×${outSize()[1]} · ${nFrames()} frames · ${(nFrames() / FPS).toFixed(1)} s`);

  const refreshInfo = () => fetchInfo().then(setInfo).catch(() => {});

  createEffect(
    () => JSON.stringify({ mode: mode(), prompt: prompt(), aspect: aspect(), mp: mp(), seconds: seconds(), steps: steps(), seed: seed(), lora: lora(), loraStrength: loraStrength(), sampler: sampler(), shiftV: shiftV(), shiftA: shiftA(), crf: crf(), fit: fit(), refSize: refSize(), mute: mute(), custom: custom(), cw: cw(), ch: ch(), upscale: upscale(), factor: factor(), split: split() }),
    (json) => {
      try {
        localStorage.setItem(STORE, json);
      } catch {}
    },
  );

  onSettled(() => {
    refreshInfo();
    const ping = () =>
      fetch("/health")
        .then((r) => setOnline(r.ok))
        .catch(() => setOnline(false));
    const timer = setInterval(ping, 8000);
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape" && preview()) setPreview(null);
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        if (!busy()) run();
      }
    };
    window.addEventListener("keydown", key);
    return () => {
      clearInterval(timer);
      window.removeEventListener("keydown", key);
    };
  });

  // ---- reference tags ----
  const picTag = (i: number) => `<Picture ${i + 1}>`;
  const vidTag = (i: number) => `<Video ${i + 1}>`;
  /** Audio tags count each reference video's soundtrack first (unless muted), then the audio files. */
  const audTag = (i: number) => `<Audio ${(mute() ? 0 : refVids().length) + i + 1}>`;

  function insert(s: string) {
    const t = untrack(prompt);
    const a = touched ? ta.selectionStart : t.length;
    const b = touched ? ta.selectionEnd : t.length;
    const ins = (a > 0 && !/\s$/.test(t.slice(0, a)) ? " " : "") + s + " ";
    setPrompt(t.slice(0, a) + ins + t.slice(b));
    const caret = a + ins.length;
    setTimeout(() => {
      ta.focus();
      ta.setSelectionRange(caret, caret);
      touched = true;
    }, 0);
  }

  async function add(kind: "image" | "video" | "audio", files: File[], put: (p: Picked[]) => void) {
    const ok = files.filter((f) => f.type.startsWith(`${kind}/`) || (kind === "video" && /\.(mp4|mov|webm|mkv)$/i.test(f.name)));
    if (!ok.length) return;
    put(await Promise.all(ok.map((f) => inspect(f, kind))));
  }

  const drop = (p: Picked | null) => p && URL.revokeObjectURL(p.url);

  // ---- generating ----
  async function request(): Promise<{ req: GenerateRequest; seed: number }> {
    const sd = seed().trim() !== "" ? Math.floor(Number(seed())) : Math.floor(Math.random() * 2 ** 32);
    const req: GenerateRequest = {
      prompt: prompt().trim(),
      seconds: seconds(),
      megapixels: mp(),
      seed: sd,
      sampler: sampler(),
      shift_video: shiftV(),
      shift_audio: shiftA(),
      crf: crf(),
    };
    if (upscaling()) {
      req.upscale = factor();
      req.upscale_split = split();
    }
    if (steps().trim() !== "") req.steps = Math.max(1, Math.floor(Number(steps())));
    if (lora()) {
      req.lora = lora();
      if (lora() !== "none") req.lora_strength = loraStrength();
    }
    const m = mode();
    if (m !== "refs" && guides().length) {
      req.guides = await Promise.all(guides().map(async (g) => ({ kind: g.kind === "audio" ? ("audio" as const) : ("image" as const), data: await dataUrl(g.p.file), frame: Math.trunc(g.frame) })));
    }
    if (m === "edit") {
      req.source = await dataUrl(src()!.file);
      req.mask = await dataUrl(maskF()!.file);
      req.source_mute = srcMute();
      req.mask_audio = maskAudio();
      if (cnet()) {
        req.controlnet = cnet();
        if (ctlVideo()) req.control_video = await dataUrl(ctlVideo()!.file);
        req.control_strength = ctlStrength();
        req.control_start = ctlRange()[0];
        req.control_end = ctlRange()[1];
      }
    }
    if (m === "frames") {
      if (first()) req.image = await dataUrl(first()!.file);
      if (last()) req.last_image = await dataUrl(last()!.file);
      if (fit()) req.fit = fit() as "stretch" | "crop";
    }
    if (m === "refs") {
      req.ref_images = await Promise.all(refImgs().map((p) => dataUrl(p.file)));
      req.ref_videos = await Promise.all(refVids().map((p) => dataUrl(p.file)));
      req.ref_audios = await Promise.all(refAuds().map((p) => dataUrl(p.file)));
      req.ref_video_mute = mute();
      req.ref_image_size = refSize();
    }
    if (custom()) {
      req.width = size()[0];
      req.height = size()[1];
    } else if (m !== "frames" || !(first() || last())) req.aspect = aspect();
    return { req, seed: sd };
  }

  async function run() {
    if (busy() || !prompt().trim()) return;
    setError("");
    setBusy(true);
    setElapsed(0);
    setSt(null);
    setRate(0);
    abort = new AbortController();
    const t0 = performance.now();
    const label = spec();
    const m = mode();
    let sampleT0 = 0;
    const tick = setInterval(() => setElapsed((performance.now() - t0) / 1000), 100);
    let live = true;
    // closing or reloading the tab mid-render loses the result: make the browser ask first
    const leave = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", leave);
    const poll = setInterval(() => {
      fetchStatus()
        .then((s) => {
          if (!live) return;
          // jobs ahead of ours (e.g. a render we stopped waiting on) own the engine: their progress isn't ours
          if (s.queued > 1) s = { ...s, state: "starting", stage: "idle", done: 0, total: 0 };
          if (s.stage === "sampling" && s.done > 0) {
            sampleT0 ||= performance.now();
            const done = s.done;
            // seconds per step since the first step finished; the first one includes model warm-up
            setRate(done > 1 ? (performance.now() - sampleT0) / 1000 / (done - 1) : 0);
          }
          setSt(s);
        })
        .catch(() => {});
    }, 450);
    try {
      const { req, seed: sd } = await request();
      const { blob, job } = await generate(req, abort.signal);
      const url = URL.createObjectURL(blob);
      const id = nextId++;
      const probe = document.createElement("video");
      const dur = await new Promise<number>((ok) => {
        probe.onloadedmetadata = () => ok(probe.duration);
        probe.onerror = () => ok(nFrames() / FPS);
        probe.src = url;
      });
      setTakes((t) => [{ id, job, url, prompt: req.prompt, seed: sd, spec: label, seconds: dur, took: (performance.now() - t0) / 1000, mode: m }, ...t]);
      setCurrent(id);
      refreshInfo();
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      clearInterval(tick);
      clearInterval(poll);
      live = false;
      window.removeEventListener("beforeunload", leave);
      setBusy(false);
      setSt(null);
      abort = null;
    }
  }

  async function removeTake(id: number) {
    const t = takes().find((x) => x.id === id);
    if (!t || !confirm("Delete this take? Its files on the server are removed too.")) return;
    if (t.job) {
      try {
        await deleteJob(t.job);
      } catch (e: any) {
        setError(`Could not delete the take: ${e.message}`);
        return;
      }
    }
    URL.revokeObjectURL(t.url);
    const rest = takes().filter((x) => x.id !== id);
    setTakes(rest);
    if (current() === id) setCurrent(rest[0]?.id ?? null);
  }

  // the upscaler runs between the two sampling passes: still "denoising" as far as the step list goes
  const stageIndex = () => STAGES.findIndex((s) => s.id === (st()?.stage === "upscaling" ? "sampling" : st()?.stage));
  const sampling = () => (st()?.stage === "sampling" ? st()! : null);
  const eta = () => {
    const s = sampling();
    return s && rate() > 0 ? Math.max(0, (s.total - s.done) * rate()) : 0;
  };

  const status = () => {
    if (!online()) return { cls: "bg-rec shadow-[0_0_8px_var(--color-rec)]", label: "Server offline" };
    if (busy()) return { cls: "bg-warn animate-pulse", label: "Rendering" };
    return { cls: "bg-ok shadow-[0_0_8px_var(--color-ok)]", label: "Engine ready" };
  };

  const modeNeedsMedia = () => (mode() === "refs" && !refImgs().length && !refVids().length && !refAuds().length) || (mode() === "frames" && !first() && !last()) || (mode() === "edit" && (!src() || !maskF()));

  return (
    <div class="flex min-h-full flex-col lg:h-full">
      {/* ---- top bar ---- */}
      <header class="flex h-12 flex-none items-center gap-4 border-b border-ink-750 bg-ink-900 px-4">
        <div class="flex items-center gap-2.5">
          <Logo size={24} />
          <span class="text-[15px] font-semibold tracking-tight text-ink-100">h3</span>
          <Show when={info()}>
            <span class="rounded bg-ink-800 px-1.5 py-px font-mono text-[10.5px] text-ink-400">v{info()!.version}</span>
          </Show>
        </div>
        <div class="flex-1" />
        <div class="flex items-center gap-2">
          <span class="inline-flex h-6 items-center gap-1.5 rounded-full border border-ink-700 bg-ink-850 px-2.5 text-xs text-ink-300">
            <span class={`${dot} ${status().cls}`} /> {status().label}
          </span>
          <Show when={info()}>
            <span class="hidden h-6 items-center gap-1.5 rounded-full border border-ink-700 bg-ink-850 px-2.5 text-xs text-ink-300 md:inline-flex" title={info()!.dit}>
              <Icon name="layers" size={12} /> {info()!.ref_dit ? "text · frames · references" : "text · frames"}
            </span>
          </Show>
          <a class={`${btnGhost} !h-7 !text-xs`} href="/docs" target="_blank">
            <Icon name="book" size={13} /> API
          </a>
        </div>
      </header>

      {/* ---- workspace ---- */}
      <div class="grid min-h-0 flex-1 gap-3 p-3 lg:grid-cols-[372px_minmax(0,1fr)_288px] lg:overflow-hidden">
        {/* compose */}
        <section class={`${panel} order-2 lg:order-1`}>
          <div class={panelHead}>
            <span class={panelTitle}>Compose</span>
            <div class="flex-1" />
            <Segmented
              class="!w-[300px]"
              value={mode()}
              onChange={(v) => setMode(v)}
              options={[
                { value: "text", label: "Text", hint: "Prompt only" },
                { value: "frames", label: "Frames", hint: "Start from a first and/or last image" },
                { value: "refs", label: "Refs", hint: "Pictures, videos and voices to use in the scene" },
                { value: "edit", label: "Edit", hint: "Regenerate part of a video under a mask, optionally steered by a Fun ControlNet" },
              ]}
            />
          </div>
          <div class="flex min-h-0 flex-1 flex-col overflow-y-auto">
            <div class="relative min-h-[200px] flex-1">
              <div class={editorBack} ref={back} aria-hidden="true">
                <For each={segments()}>
                  {(s) => (s.kind ? <span class={s.kind === "Picture" ? tokPic : s.kind === "Video" ? tokVid : tokAud}>{s.t}</span> : s.t)}
                </For>
              </div>
              <textarea
                ref={ta}
                class={editorInput}
                value={prompt()}
                spellcheck={false}
                onInput={(e) => {
                  touched = true;
                  setPrompt(e.currentTarget.value);
                }}
                onFocus={() => (touched = true)}
                onScroll={() => (back.scrollTop = ta.scrollTop)}
                placeholder="Describe the scene, and the sound that goes with it: footsteps, rain, a voice saying something specific…"
              />
            </div>

            <div class="flex flex-wrap items-center gap-1.5 border-t border-ink-750 px-3.5 py-2">
              <span class="mr-1 text-[11px] uppercase tracking-wider text-ink-500">Try</span>
              <For each={EXAMPLES}>
                {(ex, i) => (
                  <button class={`${chipBtn} !px-2`} title={ex} onClick={() => setPrompt(ex)}>
                    <Icon name="sparkle" size={11} /> {["Fox", "Violin", "Surf", "Café"][i()]}
                  </button>
                )}
              </For>
            </div>

            <Show when={mode() === "frames"}>
              <div class="border-t border-ink-750 p-3.5">
                <div class={sectionLabel}>Keyframes</div>
                <div class="grid grid-cols-2 gap-2.5">
                  <For each={[{ k: "First frame", get: first, set: setFirst }, { k: "Last frame", get: last, set: setLast }]}>
                    {(slot) => (
                      <Show
                        when={slot.get()}
                        fallback={
                          <Drop
                            accept="image/*"
                            onFiles={(f) => add("image", f, (p) => slot.set(p[0] ?? null))}
                            class="grid aspect-video place-items-center rounded-lg border border-dashed border-ink-600 text-ink-400 hover:border-ink-500 hover:text-ink-200"
                          >
                            <span class="text-center text-xs leading-snug">
                              <Icon name="image" size={18} class="mx-auto mb-1" />
                              {slot.k}
                              <span class="block text-[11px] text-ink-500">click or drop</span>
                            </span>
                          </Drop>
                        }
                      >
                        {(p) => (
                          <Thumb
                            p={p()}
                            kind="image"
                            class="aspect-video"
                            tag={slot.k}
                            onRemove={() => {
                              drop(p());
                              slot.set(null);
                            }}
                          />
                        )}
                      </Show>
                    )}
                  </For>
                </div>
                <p class="mt-2 text-xs leading-snug text-ink-500">The canvas follows the first image's aspect ratio. Use a first frame alone to animate a picture, or both to bridge them.</p>
                <Show when={first() || last()}>
                  <div class="mt-2.5">
                    <div class={sectionLabel}>Fit to canvas</div>
                    <Segmented
                      value={fit()}
                      onChange={setFit}
                      options={[
                        { value: "", label: "Default", hint: "First frame stretched, last frame cropped (as in ComfyUI)" },
                        { value: "stretch", label: "Stretch" },
                        { value: "crop", label: "Crop" },
                      ]}
                    />
                  </div>
                </Show>
              </div>
            </Show>


            <Show when={mode() === "edit"}>
              <div class="flex flex-col gap-3.5 border-t border-ink-750 p-3.5">
                <div class="grid grid-cols-2 gap-2.5">
                  <For each={[{ k: "Source video", kind: "video" as const, get: src, set: setSrc, icon: "film" }, { k: "Mask (white = redo)", kind: "video" as const, get: maskF, set: setMaskF, icon: "layers" }]}>
                    {(slot) => (
                      <Show
                        when={slot.get()}
                        fallback={
                          <Drop
                            accept="video/*,image/*"
                            onFiles={(f) => add(f[0]?.type.startsWith("image/") ? "image" : "video", f, (p) => slot.set(p[0] ?? null))}
                            class="grid aspect-video place-items-center rounded-lg border border-dashed border-ink-600 text-ink-400 hover:border-ink-500 hover:text-ink-200"
                          >
                            <span class="text-center text-xs leading-snug">
                              <Icon name={slot.icon} size={18} class="mx-auto mb-1" />
                              {slot.k}
                              <span class="block text-[11px] text-ink-500">click or drop</span>
                            </span>
                          </Drop>
                        }
                      >
                        {(p) => (
                          <Thumb
                            p={p()}
                            kind={p().file.type.startsWith("image/") ? "image" : "video"}
                            class="aspect-video"
                            tag={slot.k}
                            onRemove={() => {
                              drop(p());
                              slot.set(null);
                            }}
                          />
                        )}
                      </Show>
                    )}
                  </For>
                </div>
                <p class="text-xs leading-snug text-ink-500">
                  The mask is an image or a video: white regenerates, black keeps the source, grey is partly. The video must be at least as long as the duration.
                </p>
                <div>
                  <div class={sectionLabel}>Soundtrack</div>
                  <Segmented
                    value={maskAudio()}
                    onChange={setMaskAudio}
                    options={[
                      { value: "follow", label: "Follow mask", hint: "Regenerate the audio over the frames where the mask is set" },
                      { value: "keep", label: "Keep" },
                      { value: "generate", label: "Regenerate" },
                    ]}
                  />
                  <Toggle label="Ignore the source's audio" hint="The soundtrack is then fully generated." checked={srcMute()} onChange={setSrcMute} />
                </div>
                <Show when={(info()?.controlnets ?? []).length}>
                  <div>
                    <div class={sectionLabel}>Fun ControlNet</div>
                    <div class="relative w-full">
                      <select class={select} value={cnet()} onChange={(e) => setCnet(e.currentTarget.value)}>
                        <option value="">None</option>
                        <For each={info()?.controlnets ?? []}>{(c) => <option value={c}>{c.replace(/\.safetensors$/, "")}</option>}</For>
                      </select>
                    </div>
                    <Show when={cnet()}>
                      <div class="mt-2.5 flex flex-col gap-2.5">
                        <Show
                          when={ctlVideo()}
                          fallback={
                            <Drop accept="video/*" onFiles={(f) => add("video", f, (p) => setCtlVideo(p[0] ?? null))} class="grid h-14 place-items-center rounded-lg border border-dashed border-ink-600 text-xs text-ink-400 hover:border-ink-500 hover:text-ink-200">
                              Control video (depth, pose, edges…) — optional
                            </Drop>
                          }
                        >
                          {(p) => (
                            <Thumb p={p()} kind="video" class="aspect-video max-w-[160px]" tag="Control" onRemove={() => { drop(p()); setCtlVideo(null); }} />
                          )}
                        </Show>
                        <Slider label="Strength" value={ctlStrength()} min={0} max={2} step={0.05} digits={2} onChange={setCtlStrength} />
                        <Slider label="Starts at" value={ctlRange()[0]} min={0} max={1} step={0.05} digits={2} onChange={(v) => setCtlRange([Math.min(v, ctlRange()[1]), ctlRange()[1]])} />
                        <Slider label="Stops at" value={ctlRange()[1]} min={0} max={1} step={0.05} digits={2} onChange={(v) => setCtlRange([ctlRange()[0], Math.max(v, ctlRange()[0])])} />
                        <p class="text-xs leading-snug text-ink-500">The source and mask also go to the ControlNet as an inpainting hint.</p>
                      </div>
                    </Show>
                  </div>
                </Show>
              </div>
            </Show>

            <Show when={mode() !== "refs"}>
              <div class="border-t border-ink-750 p-3.5">
                <div class={sectionLabel}>
                  Guides
                  <span class="font-mono normal-case tracking-normal text-ink-500">{guides().length || ""}</span>
                </div>
                <div class="flex flex-col gap-2">
                  <For each={guides()}>
                    {(g, i) => (
                      <div class="flex items-center gap-2">
                        <Thumb
                          p={g.p}
                          kind={g.kind}
                          class="size-10 shrink-0"
                          onRemove={() => {
                            drop(g.p);
                            setGuides((all) => all.filter((x) => x.p.id !== g.p.id));
                          }}
                        />
                        <span class="min-w-0 flex-1 truncate text-xs text-ink-300" title={g.p.name}>{g.p.name}</span>
                        <label class="flex items-center gap-1 text-xs text-ink-500">
                          frame
                          <input
                            type="number"
                            class={`${input} !w-[72px]`}
                            value={g.frame}
                            title="Frame the guide starts at; negative counts from the end"
                            onChange={(e) => setGuides((all) => all.map((x, j) => (j === i() ? { ...x, frame: Math.trunc(Number(e.currentTarget.value)) || 0 } : x)))}
                          />
                        </label>
                      </div>
                    )}
                  </For>
                  <Drop
                    accept="image/*,video/*,audio/*"
                    multiple
                    onFiles={async (files) => {
                      for (const f of files) {
                        const kind = f.type.startsWith("audio/") ? "audio" : f.type.startsWith("image/") ? "image" : /^video\/|\.(mp4|mov|webm|mkv)$/i.test(f.type + f.name) ? "video" : null;
                        if (!kind) continue;
                        const p = await inspect(f, kind);
                        setGuides((all) => [...all, { p, kind, frame: 0 }]);
                      }
                    }}
                    class="grid h-10 place-items-center rounded-lg border border-dashed border-ink-600 text-xs text-ink-400 hover:border-ink-500 hover:text-ink-200"
                  >
                    Add a picture, clip or voice at a frame
                  </Drop>
                </div>
                <p class="mt-2 text-xs leading-snug text-ink-500">Anchors media at any frame (negative counts from the end). Clips are cut to 5, 22, 39… frames; audio is cropped to the remaining time.</p>
              </div>
            </Show>

            <Show when={mode() === "refs"}>
              <div class="flex flex-col gap-3.5 border-t border-ink-750 p-3.5">
                <For
                  each={[
                    { title: "Pictures", kind: "image" as const, accept: "image/*", get: refImgs, set: setRefImgs, tag: picTag, icon: "image" },
                    { title: "Videos", kind: "video" as const, accept: "video/*", get: refVids, set: setRefVids, tag: vidTag, icon: "film" },
                    { title: "Audio", kind: "audio" as const, accept: "audio/*", get: refAuds, set: setRefAuds, tag: audTag, icon: "music" },
                  ]}
                >
                  {(g) => (
                    <div>
                      <div class={sectionLabel}>
                        {g.title}
                        <span class="font-mono normal-case tracking-normal text-ink-500">{g.get().length || ""}</span>
                      </div>
                      <div class="grid grid-cols-4 gap-2">
                        <For each={g.get()}>
                          {(p, i) => (
                            <Thumb
                              p={p}
                              kind={g.kind}
                              class="aspect-square"
                              tag={g.tag(i())}
                              onTag={() => insert(g.tag(i()))}
                              onRemove={() => {
                                drop(p);
                                g.set((all) => all.filter((x) => x.id !== p.id));
                              }}
                            />
                          )}
                        </For>
                        <Drop
                          accept={g.accept}
                          multiple
                          onFiles={(f) => add(g.kind, f, (ps) => g.set((all) => [...all, ...ps]))}
                          class="grid aspect-square place-items-center rounded-lg border border-dashed border-ink-600 text-ink-400 hover:border-ink-500 hover:text-ink-200"
                          title={`Add ${g.title.toLowerCase()}`}
                        >
                          <Icon name={g.icon} size={17} />
                        </Drop>
                      </div>
                    </div>
                  )}
                </For>
                <p class="text-xs leading-snug text-ink-500">
                  Click a tag to put it in the prompt, e.g. <span class="font-mono text-pic">&lt;Picture 1&gt;</span> walks through the scene of{" "}
                  <span class="font-mono text-vid">&lt;Video 1&gt;</span>, speaking with the voice of <span class="font-mono text-aud">&lt;Audio 2&gt;</span>. A video's soundtrack is the
                  audio tag just before the audio files.
                </p>
                <Show when={!info()?.ref_dit}>
                  <div class={errorBox}>The reference-to-video checkpoint is not installed on this server, so references will fall back to the standard model.</div>
                </Show>
              </div>
            </Show>
          </div>

          <Show when={error()}>
            <div class={`${errorBox} mx-3 mb-2`}>
              <span class="flex-1 whitespace-pre-wrap break-words">{error()}</span>
              <button class="cursor-pointer text-ink-400 hover:text-ink-100" onClick={() => setError("")} aria-label="Dismiss">
                <Icon name="x" size={13} />
              </button>
            </div>
          </Show>
          <div class="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-ink-750 px-3.5 py-2.5">
            <span class="font-mono text-xs text-ink-400">{spec()}</span>
            <div class="flex-1" />
            <Show
              when={busy()}
              fallback={
                <button class={`${btnPrimary} ${btnLg}`} onClick={run} disabled={!prompt().trim() || !online() || modeNeedsMedia()} title={modeNeedsMedia() ? "Add the media this mode needs first" : "Ctrl+Enter"}>
                  <Icon name="bolt" size={15} /> Generate
                </button>
              }
            >
              <button class={`${btnStop} ${btnLg}`} onClick={() => { abort?.abort(); cancelRender(); }} title="Cancels the render">
                <Icon name="stop" size={13} /> Cancel · {elapsed().toFixed(0)}s
              </button>
            </Show>
          </div>
        </section>

        {/* stage + takes */}
        <div class="order-1 flex min-h-0 flex-col gap-3 lg:order-2">
          <section class={`${panel} min-h-[360px] flex-1 overflow-hidden`}>
            <div class="stage-bg relative grid min-h-0 flex-1 place-items-center">
              <Show when={shown()}>
                {(t) => (
                  <video
                    class="absolute inset-0 size-full bg-black object-contain"
                    src={t().url}
                    controls
                    autoplay
                    loop
                    playsinline
                  />
                )}
              </Show>
              <Show when={!shown() && !busy()}>
                <div class="max-w-sm px-6 text-center">
                  <div class="mx-auto mb-4 w-fit rounded-2xl p-3 shadow-[0_0_60px_color-mix(in_srgb,var(--color-accent)_30%,transparent)]">
                    <Logo size={56} />
                  </div>
                  <h2 class="text-lg font-semibold tracking-tight text-ink-100">Describe a scene, get a film clip with sound</h2>
                  <p class="mt-1.5 leading-relaxed text-ink-400">
                    Video and stereo audio come out of the same model, so mention what is heard as well as what is seen. <span class={kbd}>Ctrl</span> <span class={kbd}>↵</span> to start.
                  </p>
                </div>
              </Show>
              <Show when={busy()}>
                <div class="absolute inset-0 grid place-items-center bg-ink-950/80 p-6 backdrop-blur-sm">
                  <div class="w-full max-w-md">
                    <div class="mb-1 flex items-baseline justify-between">
                      <span class="text-sm font-medium text-ink-100">
                        {st()?.state === "starting" && (st()?.queued ?? 0) > 1 ? `Waiting for ${st()!.queued - 1} ahead of you…` : (st()?.stage === "downloading" ? "Downloading a model" : st()?.stage === "upscaling" ? "Upscaling the latent" : (STAGES[stageIndex()]?.label ?? "Starting"))}
                        …
                      </span>
                      <span class="font-mono text-xs text-ink-400">{elapsed().toFixed(1)} s</span>
                    </div>
                    <div class="relative h-2 overflow-hidden rounded-full bg-ink-800">
                      <Show
                        when={sampling()}
                        fallback={<div class="absolute inset-y-0 w-2/5 animate-sweep rounded-full bg-[linear-gradient(90deg,transparent,var(--color-accent),transparent)]" />}
                      >
                        {(s) => (
                          <div
                            class="h-full rounded-full bg-[linear-gradient(90deg,var(--color-accent-strong),var(--color-glow))] transition-[width] duration-300"
                            style={{ width: `${(s().done / Math.max(1, s().total)) * 100}%` }}
                          />
                        )}
                      </Show>
                    </div>
                    <div class="mt-1.5 flex justify-between font-mono text-[11px] text-ink-400">
                      <span>{sampling() ? `step ${sampling()!.done} / ${sampling()!.total}` : st()?.stage === "downloading" && st()!.total ? `${st()!.done} / ${st()!.total} MB` : st()?.stage === "decoding" && st()!.total ? `chunk ${st()!.done} / ${st()!.total}` : " "}</span>
                      <span>{rate() > 0 ? `${rate().toFixed(2)} s/step${eta() > 0 ? ` · ~${Math.ceil(eta())} s left` : ""}` : " "}</span>
                    </div>
                    <ol class="mt-5 grid grid-cols-4 gap-1.5">
                      <For each={STAGES}>
                        {(s, i) => (
                          <li class={`rounded-lg border px-2 py-1.5 text-center text-[11px] ${i() < stageIndex() ? "border-ok/30 text-ok" : i() === stageIndex() ? "border-accent/60 bg-accent/10 text-ink-100" : "border-ink-700 text-ink-500"}`}>
                            {i() < stageIndex() ? "✓ " : ""}
                            {s.label.split(" ")[0]}
                          </li>
                        )}
                      </For>
                    </ol>
                  </div>
                </div>
              </Show>
            </div>
            <Show when={shown()}>
              {(t) => (
                <div class="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-t border-ink-750 px-3.5 py-2.5">
                  <p class="line-clamp-1 min-w-0 flex-1 text-ink-200" title={t().prompt}>
                    {t().prompt}
                  </p>
                  <span class="font-mono text-[11px] text-ink-400">
                    {t().spec} · seed {t().seed} · rendered in {t().took.toFixed(1)} s
                  </span>
                  <div class="flex flex-none gap-1">
                    <button
                      class={`${btnGhost} ${btnSmIcon}`}
                      title="Use this prompt and seed again"
                      onClick={() => {
                        setPrompt(t().prompt);
                        setSeed(String(t().seed));
                      }}
                    >
                      <Icon name="copy" size={13} />
                    </button>
                    <a class={`${btnGhost} ${btnSmIcon}`} title="Download MP4" href={t().url} download={`h3-${t().id + 1}.mp4`}>
                      <Icon name="download" size={13} />
                    </a>
                    <button class={`${btnGhost} ${btnSmIcon}`} title="Delete take" onClick={() => removeTake(t().id)}>
                      <Icon name="trash" size={13} />
                    </button>
                  </div>
                </div>
              )}
            </Show>
          </section>

          <Show when={takes().length}>
            <section class={`${panel} flex-none`}>
              <div class={panelHead}>
                <span class={panelTitle}>Takes</span>
                <span class="font-mono text-[11px] text-ink-500">{takes().length}</span>
                <div class="flex-1" />
                <button
                  class={`${btnGhost} ${btnSm}`}
                  onClick={() => {
                    takes().forEach((t) => URL.revokeObjectURL(t.url));
                    setTakes([]);
                    setCurrent(null);
                  }}
                >
                  Clear all
                </button>
              </div>
              <div class="flex gap-2.5 overflow-x-auto p-3">
                <For each={takes()}>
                  {(t) => (
                    <button
                      class={`${take} group relative w-[148px] flex-none cursor-pointer text-left transition-colors ${current() === t.id ? "!border-accent/70 bg-accent/10" : "hover:border-ink-600"}`}
                      onClick={() => setCurrent(t.id)}
                    >
                      <video src={t.url} muted preload="metadata" class="aspect-video w-full rounded-md bg-black object-cover" />
                      <span class="mt-1.5 block truncate px-0.5 text-xs text-ink-200">{t.prompt}</span>
                      <span class="block px-0.5 font-mono text-[10.5px] text-ink-500">{t.seconds.toFixed(1)} s · {t.took.toFixed(0)} s to render</span>
                    </button>
                  )}
                </For>
              </div>
            </section>
          </Show>
        </div>

        {/* inspector */}
        <aside class={`${panel} order-3 overflow-y-auto`}>
          <div class={panelHead}>
            <span class={panelTitle}>Inspector</span>
          </div>
          <div class="flex flex-col gap-5 p-3.5">
            <div>
              <div class={sectionLabel}>
                Format
                <Show when={mode() === "frames" && (first() || last())}>
                  <span class="font-normal normal-case tracking-normal text-ink-500">from your image</span>
                </Show>
              </div>
              <div class="grid grid-cols-3 gap-1.5">
                <For each={ASPECTS}>
                  {([name, w, h]) => (
                    <button
                      class="flex h-9 cursor-pointer items-center justify-center gap-2 rounded-lg border border-ink-700 bg-ink-850 text-xs text-ink-300 transition-colors hover:border-ink-500 aria-pressed:border-accent/60 aria-pressed:bg-accent/10 aria-pressed:text-ink-100 disabled:opacity-40"
                      aria-pressed={aspect() === name ? "true" : "false"}
                      disabled={custom() || (mode() === "frames" && !!(first() || last()))}
                      onClick={() => setAspect(name)}
                    >
                      <Ratio w={w} h={h} on={aspect() === name} /> {name}
                    </button>
                  )}
                </For>
              </div>
              <Show when={!custom()}>
                <div class="mt-2.5">
                  <Segmented value={mp()} onChange={setMp} options={QUALITY} />
                </div>
              </Show>
              <Toggle label="Custom size" hint="Set the width and height yourself (snapped to multiples of 32)." checked={custom()} onChange={(v) => { if (v) { setCw(size()[0]); setCh(size()[1]); } setCustom(v); }} />
              <Show when={custom()}>
                <div class="flex items-center gap-2">
                  <input class={`${input} text-right font-mono`} type="number" min="64" max="4096" step="32" aria-label="Width in pixels" value={cw()} onInput={(e) => setCw(Number(e.currentTarget.value))} onChange={() => setCw(snap(cw()))} />
                  <span class="text-ink-500">×</span>
                  <input class={`${input} text-right font-mono`} type="number" min="64" max="4096" step="32" aria-label="Height in pixels" value={ch()} onInput={(e) => setCh(Number(e.currentTarget.value))} onChange={() => setCh(snap(ch()))} />
                </div>
              </Show>
              <p class="mt-1.5 font-mono text-xs text-ink-400">
                {size()[0]} × {size()[1]} px
              </p>
            </div>

            <div>
              <div class={sectionLabel}>Length</div>
              <Slider label="Duration" value={seconds()} min={1} max={30} step={0.5} digits={1} unit=" s" onChange={setSeconds} />
              <p class="font-mono text-xs text-ink-400">{nFrames()} frames at {FPS} fps</p>
            </div>

            <div>
              <div class={sectionLabel}>Sampling</div>
              <label class="flex items-center justify-between gap-3 py-1.5">
                <span class="text-ink-300">Steps</span>
                <input
                  class={`${input} !w-24 text-right font-mono`}
                  type="number"
                  min="1"
                  max="100"
                  placeholder={String(info()?.default_steps ?? 20)}
                  value={steps()}
                  onInput={(e) => setSteps(e.currentTarget.value)}
                />
              </label>
              <label class="flex items-center justify-between gap-3 py-1.5">
                <span class="text-ink-300">Seed</span>
                <span class="flex items-center gap-1.5">
                  <input class={`${input} !w-24 text-right font-mono`} type="number" min="0" placeholder="random" value={seed()} onInput={(e) => setSeed(e.currentTarget.value)} />
                  <button class={`${btn} ${btnIcon}`} title="Pick a random seed" onClick={() => setSeed(String(Math.floor(Math.random() * 1e6)))}>
                    <Icon name="dice" />
                  </button>
                </span>
              </label>
            </div>

            <Show when={canUpscale()}>
              <div>
                <div class={sectionLabel}>Latent upscale</div>
                <Toggle
                  label="Sample small, upscale, refine"
                  hint="The first steps run at the selected size, the learned latent upscaler enlarges the result, and the rest of the steps finish it at the larger size."
                  checked={upscale()}
                  onChange={setUpscale}
                />
                <Show when={upscale()}>
                  <Slider label="Factor" value={factor()} min={1.25} max={4} step={0.25} digits={2} unit="×" onChange={setFactor} />
                  <Slider label="Steps at selected size" value={split()} min={0.25} max={0.75} step={0.05} digits={2} onChange={setSplit} />
                  <p class="mt-1 font-mono text-xs text-ink-400">
                    sampled at {size()[0]} × {size()[1]}, delivered at {outSize()[0]} × {outSize()[1]} px
                  </p>
                </Show>
              </div>
            </Show>

            <div>
              <div class={sectionLabel}>LoRA</div>
              <div class="relative w-full">
                <select class={select} value={lora()} onChange={(e) => setLora(e.currentTarget.value)}>
                  <option value="">{info()?.default_lora ? `Server default (${info()!.default_lora!.split("/").pop()})` : "None"}</option>
                  <Show when={info()?.default_lora}>
                    <option value="none">None</option>
                  </Show>
                  <For each={info()?.loras ?? []}>{(l) => <option value={l}>{l.replace(/\.safetensors$/, "")}</option>}</For>
                </select>
                <Icon name="chevron" size={13} class="pointer-events-none absolute top-2.5 right-2.5 text-ink-400" />
              </div>
              <Show when={lora() && lora() !== "none"}>
                <Slider label="Strength" value={loraStrength()} min={0} max={2} step={0.05} digits={2} onChange={setLoraStrength} />
              </Show>
              <p class="mt-1.5 text-xs leading-snug text-ink-500">Turbo LoRAs need fewer steps: 8 for text and frames, 4 for references. A new LoRA is merged on the first request that uses it.</p>
              <div class="mt-2 flex gap-1.5">
                <button class={`${chipBtn} !px-2`} onClick={() => setSteps("8")}>8 steps</button>
                <button class={`${chipBtn} !px-2`} onClick={() => setSteps("4")}>4 steps</button>
                <button class={`${chipBtn} !px-2`} onClick={() => setSteps("")}>default</button>
              </div>
            </div>

            <div>
              <button class="flex w-full cursor-pointer items-center justify-between text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-400 hover:text-ink-200" onClick={() => setAdvanced(!advanced())} aria-expanded={advanced() ? "true" : "false"}>
                Advanced
                <Icon name="chevron" size={13} class={advanced() ? "rotate-180" : ""} />
              </button>
              <Show when={advanced()}>
                <div class="mt-2">
                  <div class="mb-1 text-ink-300">Sampler</div>
                  <Segmented
                    value={sampler()}
                    onChange={setSampler}
                    options={[
                      { value: "res_multistep", label: "res_multistep" },
                      { value: "euler", label: "euler" },
                    ]}
                  />
                  <Slider label="Video shift" value={shiftV()} min={1} max={20} step={0.5} digits={1} onChange={setShiftV} />
                  <Slider label="Audio shift" value={shiftA()} min={1} max={20} step={0.5} digits={1} onChange={setShiftA} />
                  <Slider label="H.264 CRF" value={crf()} min={10} max={32} step={1} onChange={setCrf} />
                  <Show when={mode() === "refs"}>
                    <div class="mt-1.5 mb-1 text-ink-300">Reference image size</div>
                    <Segmented
                      value={refSize()}
                      onChange={setRefSize}
                      options={[
                        { value: "match", label: "Match canvas", hint: "Scale references down to the generation's pixel area" },
                        { value: "max", label: "Max", hint: "2048 px short edge: best identity fidelity, slower" },
                      ]}
                    />
                    <Toggle label="Mute video soundtracks" hint="Use only the audio files as sound references." checked={mute()} onChange={setMute} />
                  </Show>
                  <button
                    class="mt-2 cursor-pointer text-xs text-ink-500 hover:text-ink-200"
                    onClick={() => {
                      setSampler("res_multistep");
                      setShiftV(12);
                      setShiftA(3);
                      setCrf(18);
                      setRefSize("match");
                      setMute(false);
                      setCustom(false);
                      setUpscale(false);
                      setFactor(1.5);
                      setSplit(0.5);
                    }}
                  >
                    Reset to defaults
                  </button>
                </div>
              </Show>
            </div>
          </div>
        </aside>
      </div>
      <Preview />
    </div>
  );
}
