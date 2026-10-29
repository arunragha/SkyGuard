import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    fs: {
      allow: [".."]
    },
    proxy: {
      "/stations": "http://127.0.0.1:8000",
      "/anomalies": "http://127.0.0.1:8000",
      "/series": "http://127.0.0.1:8000",
      "/settings": "http://127.0.0.1:8000",
      "/metrics": "http://127.0.0.1:8000",
      "/stream": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@shared": path.resolve(__dirname, "../shared"),
    },
  },
});
