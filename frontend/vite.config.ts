import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const here = (path: string) => fileURLToPath(new URL(path, import.meta.url));

// The offline report stays one classic script. The workbench is a normal HTTP
// bundle and shares the same React components without adding a server framework.
export default defineConfig(({ mode }) => {
  if (mode === "workbench") {
    return {
      root: here("./workbench"),
      base: "./",
      build: {
        outDir: here("../src/reward_lens/product/serve/assets"),
        emptyOutDir: true,
        target: "es2020",
        sourcemap: false,
        reportCompressedSize: false,
      },
    };
  }
  return {
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
  };
});
