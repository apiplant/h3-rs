/* Small shared building blocks: icons, the logo, a toggle, a slider and a segmented control. */
import { For, Show } from "solid-js";
import { slider, switchCls } from "./styles";

const PATHS: Record<string, string> = {
  play: "M7 4.5v15l12-7.5z",
  stop: "M6 6h12v12H6z",
  plus: "M12 5v14M5 12h14",
  download: "M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14",
  trash: "M5 7h14M10 7V5h4v2m-6 0l.7 12h6.6L16 7",
  dice: "M6 4h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM8.5 8.5h.01M15.5 8.5h.01M12 12h.01M8.5 15.5h.01M15.5 15.5h.01",
  x: "M6 6l12 12M18 6L6 18",
  check: "M5 12.5l4.5 4.5L19 7.5",
  bolt: "M13 3L5 14h6l-1 7 8-11h-6z",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  wand: "M15 4l1 2.5L18.5 7.5 16 8.5 15 11l-1-2.5-2.5-1L14 6.5zM5 19l9-9m-9 9l2-2",
  chevron: "M6 9l6 6 6-6",
  copy: "M9 9h10v11H9zM5 15V4h10",
  book: "M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3zM5 17a3 3 0 0 1 3-3h10",
  image: "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M9 9.5h.01",
  film: "M4 4h16v16H4zM8 4v16M16 4v16M4 8h4M4 12h4M4 16h4M16 8h4M16 12h4M16 16h4",
  music: "M9 18V6l10-2v12M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0zM19 16a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z",
  type: "M5 6V4h14v2M12 4v16M9 20h6",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  volume: "M4 9.5v5h4l5 4v-13l-5 4zM16.5 9a4 4 0 0 1 0 6",
  sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 17l.7 1.8L21.5 19.5l-1.8.7L19 22l-.7-1.8-1.8-.7 1.8-.7z",
};

export function Icon(props: { name: keyof typeof PATHS | string; size?: number; class?: string }) {
  return (
    <svg
      class={props.class}
      width={props.size ?? 15}
      height={props.size ?? 15}
      viewBox="0 0 24 24"
      fill={props.name === "play" || props.name === "stop" ? "currentColor" : "none"}
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[props.name] ?? ""} />
    </svg>
  );
}

export function Logo(props: { size?: number }) {
  const s = props.size ?? 22;
  return (
    <svg width={s} height={s} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="h3-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#ffb066" />
          <stop offset="1" stop-color="#ff5c8a" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="#201a17" />
      <path d="M9 8.5v15M23 8.5v15M9 16h14" stroke="url(#h3-logo)" stroke-width="2.6" fill="none" stroke-linecap="round" />
    </svg>
  );
}

export function Toggle(props: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <div class="flex items-center justify-between gap-3 py-1.5">
      <span class="min-w-0">
        <span class="block text-ink-100">{props.label}</span>
        <Show when={props.hint}>
          <span class="block text-[12px] leading-snug text-ink-400">{props.hint}</span>
        </Show>
      </span>
      <button class={switchCls} role="switch" aria-checked={props.checked ? "true" : "false"} aria-label={props.label} onClick={() => props.onChange(!props.checked)} />
    </div>
  );
}

export function Slider(props: { label: string; value: number; min: number; max: number; step: number; digits?: number; unit?: string; onChange: (v: number) => void }) {
  const fill = () => `${((props.value - props.min) / (props.max - props.min)) * 100}%`;
  return (
    <label class="block py-1.5">
      <span class="mb-0.5 flex justify-between text-ink-300">
        {props.label}
        <span class="font-mono text-[12px] text-ink-100">
          {props.value.toFixed(props.digits ?? 0)}
          {props.unit}
        </span>
      </span>
      <input
        class={slider}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        style={{ "--fill": fill() }}
        onInput={(e) => props.onChange(+e.currentTarget.value)}
      />
    </label>
  );
}

/** A row of mutually exclusive buttons. */
export function Segmented<T extends string | number>(props: {
  value: T;
  options: { value: T; label: string; hint?: string }[];
  onChange: (v: T) => void;
  class?: string;
}) {
  return (
    <div class={`flex gap-0.5 rounded-[10px] border border-ink-700 bg-ink-850 p-0.5 ${props.class ?? ""}`} role="radiogroup">
      <For each={props.options}>
        {(o) => (
          <button
            role="radio"
            aria-checked={props.value === o.value ? "true" : "false"}
            title={o.hint}
            class="h-[26px] min-w-0 flex-1 cursor-pointer whitespace-nowrap rounded-[8px] px-2 text-xs font-medium text-ink-300 transition-colors hover:text-ink-100 aria-checked:bg-ink-700 aria-checked:text-ink-100 aria-checked:shadow-sm"
            onClick={() => props.onChange(o.value)}
          >
            {o.label}
          </button>
        )}
      </For>
    </div>
  );
}

/** A little rectangle in the shape of an aspect ratio. */
export function Ratio(props: { w: number; h: number; on?: boolean }) {
  const long = 16;
  const k = long / Math.max(props.w, props.h);
  return (
    <span
      class={`inline-block rounded-[3px] border-[1.5px] ${props.on ? "border-accent" : "border-ink-500"}`}
      style={{ width: `${props.w * k}px`, height: `${props.h * k}px` }}
    />
  );
}
