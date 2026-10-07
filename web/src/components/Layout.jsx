// 可拖动布局：分隔条、可调高度的卡片、浮动 / 弹出窗口、停靠区
// Draggable layout: splitters, height-resizable cards, floating / pop-out windows and a dock area.
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { observeRoot, tr } from "../i18n/index.js";

// ---- 记住布局（存在浏览器里）/ persisted layout state ------------------------------------------
const LS = "nc.layout.v1";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(LS) || "{}"); } catch { saved = {}; }
let timer = 0;
function persist(key, value) {
  saved[key] = value;
  clearTimeout(timer);
  timer = setTimeout(() => { try { localStorage.setItem(LS, JSON.stringify(saved)); } catch { /* 忽略 */ } }, 200);
}
export function resetLayout() { try { localStorage.removeItem(LS); } catch { /* 忽略 */ } location.reload(); }

export function useStored(key, initial) {
  const [v, setV] = useState(() => (key in saved ? saved[key] : initial));
  const set = useCallback((nv) => setV((old) => {
    const val = typeof nv === "function" ? nv(old) : nv;
    persist(key, val);
    return val;
  }), [key]);
  return [v, set];
}

// ---- 拖动工具 / pointer drag helper ---------------------------------------------------------------
function startDrag(e, onMove, onEnd, cursor) {
  e.preventDefault();
  const doc = e.currentTarget.ownerDocument, win = doc.defaultView;
  const mask = doc.createElement("div");                     // 拖动时盖住 iframe/canvas，避免丢事件
  mask.className = "drag-mask";
  mask.style.cursor = cursor || "default";
  doc.body.appendChild(mask);
  const move = (ev) => onMove(ev);
  const up = (ev) => {
    win.removeEventListener("pointermove", move);
    win.removeEventListener("pointerup", up);
    mask.remove();
    onEnd?.(ev);
  };
  win.addEventListener("pointermove", move);
  win.addEventListener("pointerup", up);
}

/** 分隔条：dir="x" 左右拖，dir="y" 上下拖；onDrag 收到鼠标位置 / splitter bar */
export function Splitter({ dir = "x", onDrag, onReset, title = "拖动调整宽度" }) {
  return (
    <div className={"splitter " + dir} title={title + "（双击恢复）"} onDoubleClick={onReset}
      onPointerDown={(e) => startDrag(e, (ev) => onDrag(ev.clientX, ev.clientY), null, dir === "x" ? "col-resize" : "row-resize")}>
      <i />
    </div>
  );
}

/** 可以拖底边改变高度的卡片容器；children 可以是函数 (高度) => 元素 / height-resizable wrapper */
export function Resizable({ id, min = 80, max = 2000, children }) {
  const [h, setH] = useStored("h." + id, null);
  const ref = useRef(null);
  const onDown = (e) => {
    const y0 = e.clientY, h0 = ref.current.getBoundingClientRect().height;
    startDrag(e, (ev) => setH(Math.round(Math.max(min, Math.min(max, h0 + ev.clientY - y0)))), null, "row-resize");
  };
  return (
    <div className={"rz" + (h ? " fixed" : "")} ref={ref} style={h ? { height: h } : undefined}>
      {typeof children === "function" ? children(h) : children}
      <div className="rz-handle" title="拖动调整高度（双击恢复）" onPointerDown={onDown} onDoubleClick={() => setH(null)}><i /></div>
    </div>
  );
}

// ---- 浮动窗口 / floating window ---------------------------------------------------------------------
export function FloatWin({ id, title, rect, setRect, z, onFocus, onDock, onPopout, children }) {
  const [max, setMax] = useState(false);
  const r = max ? { x: 8, y: 8, w: window.innerWidth - 16, h: window.innerHeight - 16 } : rect;
  const move = (e) => {
    if (max) return;
    const sx = e.clientX, sy = e.clientY, r0 = rect;
    startDrag(e, (ev) => setRect({
      ...r0,
      x: Math.max(-r0.w + 80, Math.min(window.innerWidth - 80, r0.x + ev.clientX - sx)),
      y: Math.max(0, Math.min(window.innerHeight - 40, r0.y + ev.clientY - sy)),
    }), null, "move");
  };
  const resize = (e) => {
    if (max) return;
    const sx = e.clientX, sy = e.clientY, r0 = rect;
    startDrag(e, (ev) => setRect({ ...r0, w: Math.max(320, r0.w + ev.clientX - sx), h: Math.max(200, r0.h + ev.clientY - sy) }), null, "nwse-resize");
  };
  return (
    <div className="float-win card" style={{ left: r.x, top: r.y, width: r.w, height: r.h, zIndex: 100 + z }}
      onPointerDownCapture={onFocus}>
      <div className="win-bar" onPointerDown={move} onDoubleClick={() => setMax((m) => !m)}>
        <span className="win-title">{title}</span>
        <span className="win-btns" onPointerDown={(e) => e.stopPropagation()}>
          <button className="wbtn" title={max ? "还原" : "最大化"} onClick={() => setMax((m) => !m)}>{max ? "❐" : "□"}</button>
          <button className="wbtn" title="弹出到新窗口" onClick={onPopout}>↗</button>
          <button className="wbtn" title="停靠回来" onClick={onDock}>⇲</button>
        </span>
      </div>
      <div className="win-body">{children}</div>
      {!max && <div className="win-resize" onPointerDown={resize} />}
    </div>
  );
}

// ---- 弹出到浏览器新窗口 / pop out into a separate browser window --------------------------------------
export function Popout({ id, title, onClose, children }) {
  const [box, setBox] = useState(null);
  const closing = useRef(false);
  useLayoutEffect(() => {
    const w = window.open("", "neurocore-" + id, "width=900,height=640,menubar=no,toolbar=no,location=no");
    if (!w) { onClose("blocked"); return; }
    const d = w.document;
    d.title = "NeuroCore · " + tr(title);
    d.head.innerHTML = '<meta charset="utf-8">';
    for (const n of document.querySelectorAll('link[rel="stylesheet"], style')) {
      const c = n.cloneNode(true);
      if (c.tagName === "LINK") c.href = n.href;              // 绝对地址 / absolute URL
      d.head.appendChild(c);
    }
    d.documentElement.lang = document.documentElement.lang;
    d.body.className = "popout-body";
    d.body.innerHTML = "";
    const root = d.createElement("div");
    root.className = "popout-root";
    d.body.appendChild(root);
    const stop = observeRoot(d.body);
    const onUnload = () => { if (!closing.current) onClose("closed"); };
    w.addEventListener("pagehide", onUnload);
    setBox({ w, root });
    return () => { closing.current = true; stop(); w.removeEventListener("pagehide", onUnload); try { w.close(); } catch { /* 忽略 */ } };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps
  return box ? createPortal(children, box.root) : null;
}

/** 当前的标签/按钮栏（停靠区里每个面板顶部）/ dock tab strip */
export function DockBar({ panels, active, setActive, layout, setLayout, onFloat, onPopout, popped, onRecall }) {
  return (
    <div className="dock-bar">
      {panels.map((p) => (
        <button key={p.id} className={"mtab" + (layout === "tabs" && active === p.id ? " on" : "") + (layout === "split" ? " on" : "")}
          onClick={() => setActive(p.id)}>{p.title}{p.badge ? " ●" : ""}</button>
      ))}
      {popped.map((p) => (
        <button key={p.id} className="mtab ghost" title="这个面板已弹出到单独的窗口" onClick={() => onRecall(p.id)}>{p.title} ↗ · 收回</button>
      ))}
      <span className="dock-tools">
        {panels.length > 1 && (
          <button className="wbtn" title={layout === "split" ? "标签页" : "并排"} onClick={() => setLayout(layout === "split" ? "tabs" : "split")}>
            {layout === "split" ? "▭" : "◫"} <span className="wlabel">{layout === "split" ? "标签页" : "并排"}</span>
          </button>
        )}
      </span>
    </div>
  );
}

/** 每个停靠面板自己的小标题栏（浮动 / 弹出按钮）/ per-panel header with float / pop-out buttons */
export function PanelHead({ title, onFloat, onPopout }) {
  return (
    <div className="panel-head">
      <span>{title}</span>
      <span className="win-btns">
        <button className="wbtn" title="浮动" onClick={onFloat}>⧉ <span className="wlabel">浮动</span></button>
        <button className="wbtn" title="弹出到新窗口" onClick={onPopout}>↗ <span className="wlabel">弹出</span></button>
      </span>
    </div>
  );
}

/** 用于测量容器尺寸 / measure an element */
export function useSize(ref) {
  const [s, setS] = useState({ w: 0, h: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((es) => setS({ w: es[0].contentRect.width, h: es[0].contentRect.height }));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return s;
}
