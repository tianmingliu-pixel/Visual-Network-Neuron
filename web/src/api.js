// 与 Python 后端通信 / talk to the Python backend
import { useEffect, useRef, useState } from "react";

// ---- 后端地址与口令 / backend URL + access token ------------------------------------------
// 默认：和页面同一个地址（本机 python -m server 打开的页面）。
// 部署到 Vercel 时：web/dist/config.js 里的 NEUROCORE_API_BASE 指向云端后端；也可以在页面右上角「后端」里随时改，存在浏览器里。
const BUILD_BASE = (window.NEUROCORE_API_BASE || import.meta.env?.VITE_API_BASE || "").replace(/\/+$/, "");
const ls = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { v == null || v === "" ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch { /* 忽略 */ } },
};
try {                                    // 支持 ?api=https://xxx 直接指定后端 / allow ?api= in the URL
  const q = new URLSearchParams(location.search).get("api");
  if (q != null) ls.set("nc.api", q.replace(/\/+$/, ""));
} catch { /* 忽略 */ }
export const apiBase = () => { const v = ls.get("nc.api"); return v != null ? v : BUILD_BASE; };
export const buildBase = () => BUILD_BASE;
export const getToken = () => ls.get("nc.token") || "";
export const saveBackend = (base, token) => { ls.set("nc.api", base == null ? null : base.replace(/\/+$/, "")); ls.set("nc.token", token); };
export const resetBackend = () => { ls.set("nc.api", null); };

export function apiFetch(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  const t = getToken();
  if (t) headers["X-NeuroCore-Token"] = t;
  return fetch(apiBase() + path, { ...opts, headers });
}

// 解析后端返回；不是 JSON 时给出可读的原因（常见：旧版后端仍在运行，或页面不是由后端打开的）
export async function readJSON(r) {
  const text = await r.text();
  let j;
  try { j = JSON.parse(text); } catch {
    const hint = r.status === 404 || r.status === 405
      ? "后端没有这个接口——多半是旧版后端还在运行，或「后端」地址填错了。请关闭旧的 PowerShell 窗口，再运行 .\\scripts\\run.ps1 -Task ui 后刷新页面。"
      : "后端返回的不是 JSON。";
    throw new Error(`${hint}（HTTP ${r.status} ${r.statusText}）`);
  }
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

export async function getJSON(url) {
  return readJSON(await apiFetch(url));
}

export async function control(body) {
  const r = await apiFetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readJSON(r);
}

const EVENTS = ["status", "metrics", "preview", "graph", "trace", "log", "reset", "memory"];

// 订阅 SSE 实时推送；handlers 通过 ref 读取，避免重复连接
// Subscribe to the server-sent event stream; handlers are read through a ref.
export function useStream(handlers) {
  const [conn, setConn] = useState("connecting");
  const ref = useRef(handlers);
  ref.current = handlers;
  useEffect(() => {
    const es = new EventSource(apiBase() + "/api/stream");
    es.onopen = () => setConn("open");
    es.onerror = () => setConn("reconnecting");
    for (const ev of EVENTS) {
      es.addEventListener(ev, (e) => {
        let d;
        try {
          d = JSON.parse(e.data);
        } catch {
          return;
        }
        ref.current[ev]?.(d);
      });
    }
    return () => es.close();
  }, []);
  return conn;
}

export const fmt = (x, digits = 4) => {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  if (typeof x !== "number") return String(x);
  if (Number.isInteger(x) && Math.abs(x) < 1e6) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return x.toExponential(2);
  return x.toFixed(digits);
};
