# Packaging

One package, `h3-rs`, carrying the `h3` binary. Linux x86_64 packages use CUDA 13 and require an NVIDIA
GPU (Ampere or newer). Apple Silicon archives use MLX on macOS 15+. Both require `ffmpeg` and the ComfyUI MiniMax H3 model files. The `packages`,
`homebrew`, `apt` and `pacman` jobs in [`.github/workflows/release.yml`](../.github/workflows/release.yml) substitute
the `@VERSION@`, `@SHA_*@`, `@ARCH@` and `@ARCHIVE_PREFIX@` placeholders with the tag's version and the checksums of
the archives that release just built, then publish the result. None of the definitions compiles the project: they
install the binary the build jobs already produced. The `crates-io` job publishes the crate (library + binary) with
`cargo publish`.

| File | Publishes to |
| --- | --- |
| `homebrew/h3-rs.rb` | `apiplant/homebrew-tap`, as `Formula/h3-rs.rb` (Linux x86_64 and macOS arm64) |
| `pacman/PKGBUILD` | the release itself, as a `.pkg.tar.zst` asset, and `apiplant/pacman` |
| `debian/control` | the release itself, as a `.deb` asset, and `apiplant/apt` |
| `apt/apt-ftparchive.conf` | `apiplant/apt`, served at `apt.apiplant.com` |

The repositories and secrets are the shared `apiplant` ones (`HOMEBREW_TAP_TOKEN`, `CARGO_REGISTRY_TOKEN`,
`PACMAN_REPO_TOKEN` + `PACMAN_GPG_PRIVATE_KEY`/`PACMAN_GPG_PASSPHRASE`, `APT_REPO_TOKEN` +
`APT_GPG_PRIVATE_KEY`/`APT_GPG_PASSPHRASE`). A publish job whose credential is absent is skipped rather than failing
the release.

## Builds

- `ui`: `pnpm build` in `ui/`, uploaded as the `ui-dist` artifact. `binary`, `binary-arch` and `crates-io` download it
  into `ui/dist` before building, and `H3_REQUIRE_UI=1` makes build.rs fail instead of embedding no web UI. The crate
  ships `ui/dist`, so `cargo install` gets the UI without Node. The tokenizer (`assets/tokenizer.json`) is compiled in.
- `binary`: an `nvidia/cuda:13.0.2-devel-ubuntu22.04` container (nvcc and the CUDA libraries, no GPU needed),
  `cargo build --profile dist --no-default-features --features serve`. candle compiles its kernels for one compute
  capability, so the release uses `CUDA_COMPUTE_CAP=80` (PTX that newer GPUs JIT-compile); flash-attn is left out, the
  engine has its own int8 attention. The archive, the `.deb` and the Homebrew formula use it.
- `binary-arch`: the same build inside Arch against Arch's own `cuda`, used by the pacman package (a different soname).
  If rolling Arch breaks that build, the pacman package falls back to the generic archive.

`.cargo/config.toml` pins `CUDARC_CUDA_VERSION` to the toolkit on the author's machine; the CI jobs override it with
the version of the toolkit they build against.

The `.deb` depends on `libcuda1` and `ffmpeg`; the CUDA 13 libraries (`libcudart`, `libcublas`, `libnvrtc`, `libcurand`)
come from NVIDIA's own repository and are not expressible as Debian dependencies. The Arch package depends on
`nvidia-utils`, `cuda` and `ffmpeg`. Homebrew cannot depend on a driver or CUDA.

## Cutting a release

Bump `version` in `Cargo.toml`, commit, then:

```bash
git tag v$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2) && git push origin --tags
```

`workflow_dispatch` runs the build and packaging jobs without publishing anything (a dry run).

## Using the packages

```bash
brew tap apiplant/tap && brew install apiplant/tap/h3-rs
```

Arch Linux (one-time setup, then `pacman -Syu` picks up new releases):

```bash
curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\n[apiplant]\nSigLevel = Required DatabaseOptional\nServer = https://apiplant.github.io/pacman/$arch\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy h3-rs
```

Debian/Ubuntu (one-time setup, then `apt upgrade` picks up new releases):

```bash
curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install h3-rs
```

Then put the ComfyUI MiniMax H3 files in a `models/` folder and run `H3_MODELS_DIR=~/ComfyUI/models h3 serve`.

## Apple Silicon archives

The native `macos-15` job follows the `fluid-speech-rs` packaging convention: a target-named
`h3-rs-vVERSION-aarch64-apple-darwin.tar.gz` with a matching `.sha256` file, CLI `--version` smoke test,
and Homebrew `on_macos` / `on_arm` URL and checksum. It also runs the MLX library tests. Linux CUDA,
Debian, Arch and repository publication retain their existing paths.

MLX's native library is statically linked, but GPU execution also requires `mlx.metallib`. The job sets
`MLX_RS_METAL_PATH` to a workspace directory and packages its generated `mlx.metallib` alongside `h3`.
The runtime resolves the adjacent file before initializing Metal; the formula instead installs it under
`share/h3-rs`. This avoids depending on the build runner's home directory after installation.
