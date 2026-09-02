import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

// The offline report, target 1 of D-68. A file:// document has an opaque origin, so an external
// module, a worker and a fetch are all unavailable to it; `format: "iife"` is what keeps the
// emitted chunk a classic script, and vite-plugin-singlefile is the check that nothing was left
// outside the file. CSS is not minified: the stylesheet of a certificate should stay readable, and
// it is a rounding error against the record.
export default defineConfig({
  root: here("./report"),
  base: "./",
  plugins: [viteSingleFile({ removeViteModuleLoader: true, deleteInlinedFiles: false })],
  build: {
    outDir: here("./dist"),
    emptyOutDir: true,
    target: "es2020",
    cssCodeSplit: false,
    cssMinify: false,
    assetsInlineLimit: 100000000,
    modulePreload: false,
    sourcemap: false,
    minify: true,
    reportCompressedSize: false,
    rollupOptions: {
      input: here("./report/index.html"),
      output: {
        format: "iife",
        inlineDynamicImports: true,
        entryFileNames: "report.js",
        assetFileNames: "report[extname]",
      },
    },
  },
});
