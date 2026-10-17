import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";
import tailwindcss from "@tailwindcss/vite";

// `pnpm dev` proxies the API to a running `h3 serve` (H3_URL, default 127.0.0.1:8188).
const api = process.env.H3_URL ?? "http://127.0.0.1:8188";

export default defineConfig({
  plugins: [solid(), tailwindcss()],
  server: { port: 5278, proxy: { "/v1": api, "/health": api, "/openapi.json": api } },
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
});
