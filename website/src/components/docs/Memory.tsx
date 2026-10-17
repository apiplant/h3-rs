import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";

export function DocsMemory() {
  return (
    <DocsLayout>
      <H1>Memory presets</H1>
      <Lead>
        Choose a preset with <IC>--preset</IC>, load a TOML file with <IC>--config</IC>, or override single knobs. <IC>h3 presets</IC> prints them.
      </Lead>

      <Section>
        <H2>Presets</H2>
        <div class="mt-4 max-w-4xl overflow-x-auto rounded-xl border border-line">
          <table class="w-full min-w-[40rem] border-collapse text-left text-sm">
            <thead>
              <tr class="bg-surface text-faint">
                <th class="border-b border-line px-4 py-2.5 font-medium">preset</th>
                <th class="border-b border-line px-4 py-2.5 font-medium">target</th>
                <th class="border-b border-line px-4 py-2.5 font-medium">DiT blocks on GPU</th>
                <th class="border-b border-line px-4 py-2.5 font-medium">TE layers on GPU</th>
                <th class="border-b border-line px-4 py-2.5 font-medium">the rest</th>
              </tr>
            </thead>
            <tbody class="text-muted">
              {[
                ["gb24 (default)", "24 GB GPU, ≥48 GB RAM", "auto", "50 / 50", "pinned RAM"],
                ["gb16", "16 GB, ≥32 GB RAM", "14", "30", "RAM"],
                ["gb12", "12 GB, ≥32 GB RAM", "4", "16", "RAM"],
                ["gb8", "8 GB, ≥16 GB RAM", "0", "4", "disk (mmap); measured peak 6.5 GB at 1344×768×124"],
                ["minimal", "smallest GPU", "0", "0", "disk; VAE decoder streamed, small chunks"],
                ["auto", "anything", "fits free VRAM", "fits free VRAM", "RAM; reserve_gb kept for activations"],
              ].map((r) => (
                <tr class="even:bg-surface/50">
                  <td class="border-b border-line px-4 py-2.5 font-mono text-[0.8125rem] text-accent">{r[0]}</td>
                  <td class="border-b border-line px-4 py-2.5">{r[1]}</td>
                  <td class="border-b border-line px-4 py-2.5">{r[2]}</td>
                  <td class="border-b border-line px-4 py-2.5">{r[3]}</td>
                  <td class="border-b border-line px-4 py-2.5">{r[4]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <P>
          Streamed weights upload while the previous layer computes, so at large sizes the low-VRAM presets cost little as long as RAM (or the page cache)
          holds the weights. If RAM is short, disk reads dominate. <IC>h3 serve</IC> keeps every model loaded: it streams the text encoder and VAE decoder and gives the DiT the rest.
        </P>
      </Section>

      <Section>
        <H2>Knobs</H2>
        <FlagTable
          rows={[
            { flag: "dit_gpu_blocks, te_gpu_layers", meaning: "Resident layers; the rest upload every step (DiT ≈0.39 GB per block, text encoder ≈0.31 GB per layer). -1 is auto." },
            { flag: 'offload = "ram" | "disk"', meaning: "ram copies non-resident weights into process memory (most RAM, steady speed); disk reads them from the mmapped file each use (least RAM, fast while the page cache holds them)." },
            { flag: "sequential", meaning: "Free each model before the next stage loads (text encoder, then DiT, then VAEs)." },
            { flag: "token_chunk, mlp_chunk", meaning: "Rows per GEMM or MLP chunk; this bounds activation memory at large sizes." },
            { flag: "vae_tile_batch, vae_resident, vae_dtype", meaning: "The VAE decodes all 256 px tiles of a 17-frame chunk layer by layer, so a streamed decoder costs one weight pass per chunk." },
          ]}
        />
      </Section>

      <Section>
        <H2>Environment switches</H2>
        <UL>
          <LI><IC>H3_INT8=0</IC> uses bf16 GEMMs; <IC>H3_GEMM=cublaslt</IC> uses cuBLASLt int8 with a separate epilogue; <IC>H3_ROT=gemm</IC> does the rotation separately.</LI>
          <LI><IC>H3_ATTN=flash</IC> uses flash-attn (when built with it); <IC>H3_LORA_RUNTIME=1</IC> keeps LoRAs unmerged as a low-rank path.</LI>
          <LI><IC>H3_PROFILE=1</IC> prints per-op times each step; <IC>H3_TOKENIZER</IC> loads a tokenizer file instead of the embedded one.</LI>
        </UL>
      </Section>
    </DocsLayout>
  );
}
