import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";
import { getLang, observeRoot } from "./i18n/index.js";

document.documentElement.lang = getLang() === "en" ? "en" : "zh-CN";
observeRoot(document.body);          // 英文模式下自动翻译界面文字 / translate UI text in English mode

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
