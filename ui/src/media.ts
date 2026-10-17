/* Browser-side helpers: files to base64, thumbnails, and the same canvas arithmetic the engine uses. */

export const FPS = 24;

export function dataUrl(f: Blob): Promise<string> {
  return new Promise((ok, err) => {
    const r = new FileReader();
    r.onload = () => ok(r.result as string);
    r.onerror = () => err(r.error);
    r.readAsDataURL(f);
  });
}

export const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const r32 = (v: number) => Math.max(32, Math.round(v / 32) * 32);

/** The canvas `h3` picks for an aspect ratio at a total area (ComfyUI's ImageScaleToTotalPixels, step 32). */
export function canvas(aw: number, ah: number, megapixels: number): [number, number] {
  const s = Math.sqrt((megapixels * 1024 * 1024) / (aw * ah));
  return [r32(aw * s), r32(ah * s)];
}

/** Frame count the model really renders: the next one on the 17k+5 grid. */
export function frames(seconds: number): number {
  let n = Math.max(5, Math.round(seconds * FPS));
  while (n % 17 !== 5) n++;
  return n;
}

export interface Picked {
  id: number;
  name: string;
  file: File;
  /** object URL for previews */
  url: string;
  /** natural size, images and videos */
  w?: number;
  h?: number;
  seconds?: number;
}

let nextId = 1;

/** Reads a picked file's natural size / duration so the UI can preview it. */
export function inspect(file: File, kind: "image" | "video" | "audio"): Promise<Picked> {
  const url = URL.createObjectURL(file);
  const base: Picked = { id: nextId++, name: file.name, file, url };
  return new Promise((ok) => {
    if (kind === "image") {
      const i = new Image();
      i.onload = () => ok({ ...base, w: i.naturalWidth, h: i.naturalHeight });
      i.onerror = () => ok(base);
      i.src = url;
    } else {
      const m = document.createElement(kind === "video" ? "video" : "audio");
      m.preload = "metadata";
      m.onloadedmetadata = () => ok({ ...base, seconds: m.duration, w: (m as HTMLVideoElement).videoWidth, h: (m as HTMLVideoElement).videoHeight });
      m.onerror = () => ok(base);
      m.src = url;
    }
  });
}
