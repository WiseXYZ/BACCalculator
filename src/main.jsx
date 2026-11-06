import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Fonts bundled into the app itself: no request ever leaves the phone.
// only the weights the stylesheet actually uses (measured in a real browser)
import "@fontsource/bricolage-grotesque/latin-800.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-600.css";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";

import App from "./App.jsx";

// The app persists through window.storage, which exists inside Claude but not
// on a phone. Without this adapter every save would quietly fail and your
// profile and tonight's drinks would vanish whenever the app was closed.
// Backed by the WebView's localStorage, which lives in the app's private data
// on the device and is never sent anywhere.
if (!window.storage) {
  const PREFIX = "bloodalc:";
  const ls = window.localStorage;
  window.storage = {
    async get(key) {
      const v = ls.getItem(PREFIX + key);
      return v === null ? null : { key, value: v, shared: false };
    },
    async set(key, value) {
      ls.setItem(PREFIX + key, String(value));
      return { key, value, shared: false };
    },
    async delete(key) {
      ls.removeItem(PREFIX + key);
      return { key, deleted: true, shared: false };
    },
    async list(prefix = "") {
      const keys = [];
      for (let i = 0; i < ls.length; i++) {
        const k = ls.key(i);
        if (k && k.startsWith(PREFIX + prefix)) keys.push(k.slice(PREFIX.length));
      }
      return { keys, prefix, shared: false };
    },
  };
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
