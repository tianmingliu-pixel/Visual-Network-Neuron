// 界面语言（中文 / English）/ UI language
// 做法：界面照常用中文写；切到英文时，一个“翻译层”监视页面上的文字节点和 title/placeholder 属性，
// 按词典替换成英文（后端发来的提示、日志、任务说明也一起翻译）。画在 canvas 上的文字用 tr() 翻译。
// Approach: the UI is authored in Chinese; in English mode a DOM translation layer swaps text nodes and
// title/placeholder attributes using the dictionary (backend messages included). Canvas text uses tr().
import { useSyncExternalStore } from "react";
import { EXACT, PATTERNS } from "./en.js";

const CJK = /[㐀-鿿]/;
const KEY = "nc.lang";
let lang = (() => {
  try {
    const q = new URLSearchParams(location.search).get("lang");
    if (q === "en" || q === "zh") return q;
    return localStorage.getItem(KEY) || (navigator.language?.startsWith("zh") ? "zh" : "en");
  } catch { return "zh"; }
})();
const listeners = new Set();

export const getLang = () => lang;
export function setLang(l) {
  if (l === lang) return;
  lang = l;
  try { localStorage.setItem(KEY, l); } catch { /* 忽略 */ }
  document.documentElement.lang = l === "en" ? "en" : "zh-CN";
  for (const root of roots) l === "en" ? translateTree(root) : restoreTree(root);
  listeners.forEach((f) => f());
}
export function useLang() {
  return useSyncExternalStore((f) => { listeners.add(f); return () => listeners.delete(f); }, () => lang);
}

// ---- 词典查找 / lookup -------------------------------------------------------------------
const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const COMPILED = PATTERNS
  .map(([zh, en]) => [new RegExp("^" + zh.split("{}").map(esc).join("([\\s\\S]+?)") + "$"), en, zh.length])
  .sort((a, b) => b[2] - a[2]);
const cache = new Map();

function lookup(s, depth = 0) {
  if (s in EXACT) return EXACT[s];
  for (const [re, en] of COMPILED) {
    const m = s.match(re);
    if (m) return en.replace(/\{(\d)\}/g, (_, i) => (depth < 3 ? trCore(m[+i + 1], depth + 1) : m[+i + 1]));
  }
  return null;
}

function trCore(s, depth = 0) {
  if (!s || !CJK.test(s)) return s;
  const lead = s.match(/^\s*/)[0], trail = s.match(/\s*$/)[0];
  const core = s.trim().replace(/\s+/g, (w) => (w.includes("\n") ? w : " "));
  let out = lookup(core, depth);
  if (out == null && core.includes("\n")) {
    const parts = core.split("\n").map((p) => trCore(p, depth + 1));
    out = parts.join("\n");
  }
  if (out == null) return s;
  return lead + out + trail;
}

/** 翻译一段文字（当前是中文模式时原样返回）/ translate a string in English mode */
export function tr(s) {
  if (lang !== "en" || typeof s !== "string" || !CJK.test(s)) return s;
  if (cache.has(s)) return cache.get(s);
  const out = trCore(s);
  if (cache.size > 5000) cache.clear();
  cache.set(s, out);
  return out;
}

// ---- DOM 翻译层 / DOM translation layer ----------------------------------------------------
const ATTRS = ["title", "placeholder", "aria-label"];
const origText = new WeakMap();   // 文字节点 → 原文 / text node → original
const setText = new WeakMap();    // 文字节点 → 我们写入的译文 / text node → what we wrote
const origAttr = new WeakMap();   // 元素 → {attr: 原文}
const roots = new Set();
const SKIP = "[data-no-i18n], script, style, textarea, code.code, .cmt, .code, .rec";

function skipped(el) { return el && el.closest && el.closest(SKIP); }

function doText(node) {
  const v = node.nodeValue;
  if (setText.get(node) === v) return;               // 我们自己刚写的 / our own write
  if (!CJK.test(v)) { origText.delete(node); return; }
  if (skipped(node.parentElement)) return;
  const t = tr(v);
  origText.set(node, v);
  if (t !== v) { setText.set(node, t); node.nodeValue = t; }
}

function doAttrs(el) {
  if (skipped(el)) return;
  for (const a of ATTRS) {
    const v = el.getAttribute(a);
    if (!v) continue;
    const rec = origAttr.get(el) || {};
    if (rec[a] && rec[a].en === v) continue;
    if (!CJK.test(v)) continue;
    const t = tr(v);
    rec[a] = { zh: v, en: t };
    origAttr.set(el, rec);
    if (t !== v) el.setAttribute(a, t);
  }
}

function walk(root, fnText, fnEl) {
  if (root.nodeType === 3) return fnText(root);
  if (root.nodeType !== 1 && root.nodeType !== 11) return;
  if (root.nodeType === 1) { if (root.matches(SKIP)) return; fnEl(root); }
  const w = (root.ownerDocument || root).createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT, {
    acceptNode: (n) => (n.nodeType === 1 && n.matches(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  let n;
  while ((n = w.nextNode())) n.nodeType === 3 ? fnText(n) : fnEl(n);
}

function translateTree(root) { walk(root, doText, doAttrs); }
function restoreTree(root) {
  walk(root, (n) => {
    const o = origText.get(n);
    if (o != null && setText.get(n) === n.nodeValue) { setText.delete(n); n.nodeValue = o; }
  }, (el) => {
    const rec = origAttr.get(el);
    if (!rec) return;
    for (const a in rec) if (el.getAttribute(a) === rec[a].en) el.setAttribute(a, rec[a].zh);
    origAttr.delete(el);
  });
}

/** 开始翻译某个根节点（主页面 / 弹出窗口）/ start translating a root (main page or a pop-out window) */
export function observeRoot(root) {
  if (roots.has(root)) return () => {};
  roots.add(root);
  if (lang === "en") translateTree(root);
  const W = root.ownerDocument?.defaultView || window;
  const mo = new W.MutationObserver((muts) => {
    if (lang !== "en") return;
    for (const m of muts) {
      if (m.type === "characterData") doText(m.target);
      else if (m.type === "attributes") doAttrs(m.target);
      else for (const n of m.addedNodes) walk(n, doText, doAttrs);
    }
  });
  mo.observe(root, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  return () => { mo.disconnect(); roots.delete(root); };
}
