import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend (web/app.py) serves the built SPA from web/frontend_dist/.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", ws: true, changeOrigin: true },
    },
  },
  build: {
    outDir: "../frontend_dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
