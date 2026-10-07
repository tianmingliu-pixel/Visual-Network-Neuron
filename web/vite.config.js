import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发模式：前端 5173 端口，/api 请求代理到 Python 后端 8765
// Dev: UI on :5173, /api proxied to the Python backend on :8765
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: true } },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
