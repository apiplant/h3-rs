/**
 * The published release and its assets.
 *
 * `release.yml` builds one archive per target and names it
 * `h3-rs-<tag>-<target>.tar.gz` (Linux x86_64 with CUDA, macOS arm64 with MLX), so a direct link
 * can be assembled here rather than sending every reader to the releases
 * page to hunt for their own. The version comes from the crate manifest at
 * build time, which means a site deployed from a commit predating the tag
 * would link at an asset that does not exist yet — deploy the site after the
 * release, not before.
 */

import { GITHUB_URL } from "./links";

export const VERSION = __VERSION__;
/** Git tags are `v`-prefixed, and the tag is part of every asset name. */
export const TAG = `v${VERSION}`;

export const RELEASES_URL = `${GITHUB_URL}/releases`;
export const LATEST_RELEASE_URL = `${RELEASES_URL}/tag/${TAG}`;

export interface Platform {
  /** Rust target triple, as it appears in the asset name. */
  target: string;
  label: string;
  /** For the download button, which cannot wrap and sits in a narrow column. */
  short: string;
  cuda?: boolean;
}

/* Exactly the targets release.yml builds: CUDA on Linux, MLX on Apple Silicon. */
export const PLATFORMS: Platform[] = [
  { target: "x86_64-unknown-linux-gnu", label: "Linux · x86_64 (CUDA)", short: "Linux x86_64, CUDA" },
  { target: "aarch64-apple-darwin", label: "macOS · Apple Silicon (MLX)", short: "macOS arm64, MLX" },
];

export function assetName(platform: Platform): string {
  const prefix = "h3-rs";
  return `${prefix}-${TAG}-${platform.target}.tar.gz`;
}

export function downloadUrl(platform: Platform): string {
  return `${RELEASES_URL}/download/${TAG}/${assetName(platform)}`;
}

export const CUDA_PLATFORM: Platform = {
  target: "x86_64-unknown-linux-gnu",
  label: "Linux · x86_64 (CUDA)",
  short: "Linux x86_64, CUDA",
  cuda: true,
};
