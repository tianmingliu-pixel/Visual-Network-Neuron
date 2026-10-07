// 轻量 Canvas 折线图：按像素分桶抽稀（每像素只画最小/最大值），几万个点也不卡
// Lightweight canvas chart with per-pixel min/max decimation — stays smooth with 10k+ points.
import { useEffect, useRef, useState } from "react";
import { fmt } from "../api.js";

function ema(values, alpha) {
  const out = new Array(values.length);
  let m = null;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v == null) { out[i] = m; continue; }
    m = m == null ? v : alpha * v + (1 - alpha) * m;
    out[i] = m;
  }
  return out;
}

export default function LineChart({ title, dataRef, version, series, logScale = false, height = 190 }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [log, setLog] = useState(logScale);
  const [hover, setHover] = useState(null);
  const [width, setWidth] = useState(400);
  const geom = useRef(null);

  useEffect(() => {
    const ro = new ResizeObserver((es) => setWidth(Math.max(200, es[0].contentRect.width)));
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr;
    cv.height = height * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const grid = css.getPropertyValue("--grid").trim() || "#333";
    const muted = css.getPropertyValue("--muted").trim() || "#888";
    ctx.clearRect(0, 0, width, height);

    const pts = dataRef.current;
    const pad = { l: 52, r: 10, t: 8, b: 22 };
    const W = width - pad.l - pad.r, H = height - pad.t - pad.b;
    const active = series.filter((s) => pts.some((p) => p[s.key] != null));
    if (!pts.length || !active.length) {
      ctx.fillStyle = muted;
      ctx.font = "12px system-ui";
      ctx.fillText("等待训练数据… / waiting for data", pad.l, height / 2);
      geom.current = null;
      return;
    }
    const xs = pts.map((p) => p.step);
    const x0 = xs[0], x1 = Math.max(xs[xs.length - 1], x0 + 1);
    const tf = (v) => (log ? Math.log10(Math.max(v, 1e-8)) : v);
    const lines = active.map((s) => {
      const raw = pts.map((p) => (p[s.key] == null ? null : tf(p[s.key])));
      return { ...s, vals: s.smooth ? ema(raw, s.smooth) : raw };
    });
    let lo = Infinity, hi = -Infinity;
    for (const ln of lines) for (const v of ln.vals) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const span = hi - lo; lo -= span * 0.05; hi += span * 0.05;
    const X = (s) => pad.l + ((s - x0) / (x1 - x0)) * W;
    const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * H;

    // 网格与坐标 / grid & axes
    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.fillStyle = muted; ctx.font = "11px system-ui";
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4, y = Y(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + W, y); ctx.stroke();
      ctx.fillText(fmt(log ? 10 ** v : v, 3), 4, y + 4);
    }
    ctx.fillText(String(x0), pad.l, height - 6);
    const lbl = String(xs[xs.length - 1]);
    ctx.fillText(lbl, pad.l + W - ctx.measureText(lbl).width, height - 6);

    // 每像素一个桶：画 min-max 竖线 + 均值连线 / one bucket per pixel
    for (const ln of lines) {
      const buckets = new Map();
      for (let i = 0; i < pts.length; i++) {
        const v = ln.vals[i]; if (v == null) continue;
        const px = Math.round(X(xs[i]));
        const b = buckets.get(px);
        if (b) { b.min = Math.min(b.min, v); b.max = Math.max(b.max, v); b.last = v; }
        else buckets.set(px, { min: v, max: v, last: v });
      }
      ctx.strokeStyle = ln.color; ctx.globalAlpha = ln.alpha ?? 1; ctx.lineWidth = ln.width ?? 1.5;
      ctx.beginPath();
      let first = true;
      for (const [px, b] of buckets) {
        if (first) { ctx.moveTo(px, Y(b.last)); first = false; }
        ctx.lineTo(px, Y(b.min)); ctx.lineTo(px, Y(b.max)); ctx.lineTo(px, Y(b.last));
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    geom.current = { pad, W, x0, x1, lines, xs };
  }, [version, width, height, log, series, dataRef]);

  const onMove = (e) => {
    const g = geom.current; if (!g) return;
    const r = canvasRef.current.getBoundingClientRect();
    const step = g.x0 + ((e.clientX - r.left - g.pad.l) / g.W) * (g.x1 - g.x0);
    let lo = 0, hi = g.xs.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (g.xs[m] < step) lo = m + 1; else hi = m; }
    const p = dataRef.current[lo];
    if (p) setHover({ x: e.clientX - r.left, p });
  };

  const last = dataRef.current[dataRef.current.length - 1];
  return (
    <div className="card chart" ref={wrapRef}>
      <div className="card-head">
        <span>{title}</span>
        <div className="legend">
          {series.map((s) => (
            <span key={s.key + (s.smooth || "")}><i style={{ background: s.color }} />{s.label}
              {last && last[s.key] != null && !s.smooth ? <b> {fmt(last[s.key])}</b> : null}</span>
          ))}
          <button className={"chip" + (log ? " on" : "")} onClick={() => setLog(!log)} title="对数坐标 / log scale">log</button>
        </div>
      </div>
      <div className="chart-body" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        <canvas ref={canvasRef} style={{ width: "100%", height }} />
        {hover && (
          <div className="tip" style={{ left: Math.min(hover.x + 12, width - 150) }}>
            <div>step {hover.p.step}</div>
            {series.filter((s) => !s.smooth && hover.p[s.key] != null).map((s) => (
              <div key={s.key}><i style={{ background: s.color }} />{s.label}: {fmt(hover.p[s.key])}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
