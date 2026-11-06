import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Content Security Policy, production builds only (the dev server needs a
// websocket for live reload, which this would rightly block).
// The privacy guarantee is connect-src 'none': no code in the page can send
// anything anywhere, by any means - fetch, XHR, websockets or beacons.
// Every other resource must come from the app's own bundled files.
// 'unsafe-inline' is present only as a safety margin for Capacitor, which
// injects its native bridge as an inline script; it does NOT permit loading
// any script from outside the app.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "font-src 'self' data:",
  "img-src 'self' data:",
  "connect-src 'none'",
  "media-src 'none'",
  "object-src 'none'",
  "frame-src 'none'",
  "worker-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");

const cspPlugin = {
  name: "inject-csp",
  apply: "build",
  transformIndexHtml(html) {
    return html.replace(
      '<meta charset="UTF-8" />',
      `<meta charset="UTF-8" />\n    <meta http-equiv="Content-Security-Policy" content="${CSP}" />`
    );
  },
};

export default defineConfig({
  plugins: [react(), cspPlugin],
  base: "./",
  // single bundle: the module-preload polyfill would be the only fetch() in the
  // app, and it has nothing to preload, so it is switched off
  build: { outDir: "dist", sourcemap: false, modulePreload: { polyfill: false } },
});
