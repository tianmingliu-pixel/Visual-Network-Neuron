import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { resolveBackend } from "./api.js";
import "./styles.css";
import { getLang, observeRoot } from "./i18n/index.js";

document.documentElement.lang = getLang() === "en" ? "en" : "zh-CN";
observeRoot(document.body);          // 英文模式下自动翻译界面文字 / translate UI text in English mode

// 先决定连哪个后端（最多等 3 秒），再渲染 / pick the backend first, then render
resolveBackend().finally(() => createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
));
