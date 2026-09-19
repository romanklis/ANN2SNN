import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

// The bundle is served by the Flask backend (server/app.py) at "/", so we emit
// relative asset URLs ("base: './'") which work at the site root *and* under a
// reverse-proxy prefix. Output goes to ../server/static.
export default defineConfig({
  root: here,
  base: "./",
  build: {
    outDir: resolve(here, "..", "server", "static"),
    emptyOutDir: true,
    target: "es2020",
    sourcemap: false,
    chunkSizeWarningLimit: 2048,
  },
});
