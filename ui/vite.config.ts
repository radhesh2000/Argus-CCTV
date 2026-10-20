import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Where `argus api` is listening during development. In production the SPA is
// served *by* the API, so every path below is same-origin.
//
// There is no `/gateway` proxy any more, and its absence is the Phase-5 change:
// go2rtc's HTTP API binds to loopback on the appliance and the console reaches
// signalling and posters through `/api/gateway/*`, which is authenticated. One
// door, and in development it is the same door.
const API = process.env.ARGUS_API_URL ?? 'http://127.0.0.1:8080';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Evidence is served by the API from static mounts; the SSE stream needs
      // no special handling beyond not being buffered, which vite does not do.
      // `ws: true` is for the WebRTC signalling relay under /api/gateway/ws.
      '/api': { target: API, changeOrigin: true, ws: true },
      '/snapshots': { target: API, changeOrigin: true },
      '/clips': { target: API, changeOrigin: true },
      '/tracks': { target: API, changeOrigin: true },
      '/index': { target: API, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    // Nothing is fetched from the internet at runtime — fonts included — so the
    // whole bundle has to be on disk. Assets stay under /assets for the API's
    // static mount to serve with a suffix (see SpaFiles in argus/api/app.py).
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 900,
  },
});
