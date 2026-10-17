# h3-rs website

The website for [h3-rs](https://github.com/apiplant/h3-rs). Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare.

```bash
pnpm install
pnpm dev       # local dev server
pnpm build     # -> dist/
pnpm check     # types only
```

The version shown on the site is read from `../Cargo.toml` at build time.
