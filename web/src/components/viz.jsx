// 小型可视化组件 / small reusable charts (SVG / canvas, no dependencies)
import { useEffect, useRef } from "react";
import { fmt } from "../api.js";
import { tr, useLang } from "../i18n/index.js";

const BASE = ["#f7768e", "#9ece6a", "#7aa2f7", "#e0af68", "#bb9af7", "#7dcfff", "#ff9e64", "#73daca", "#c0caf5", "#db4b4b"];
export const classColor = (i) => (i < BASE.length ? BASE[i] : `hsl(${(i * 47) % 360} 65% 62%)`);

// 图片缩略图：支持灰度(c=1)与彩色(c=3) / thumbnail for gray or RGB pixel arrays
export function Thumb({ w, h, c = 1, px, scale = 2, title }) {
  const ref = useRef(null);
  useEffect(() => {
    const ctx = ref.current.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const r = c === 3 ? px[i * 3] : px[i], g = c === 3 ? px[i * 3 + 1] : px[i], b = c === 3 ? px[i * 3 + 2] : px[i];
      img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [w, h, c, px]);
  return <canvas ref={ref} width={w} height={h} title={title} className="px" style={{ width: w * scale, height: h * scale }} />;
}

// 横向条形图 / horizontal bars
export function Bars({ items, max, color = "#7aa2f7", format = (v) => fmt(v, 3), labelWidth = 150 }) {
  const m = max ?? Math.max(1e-12, ...items.map((d) => Math.abs(d.value)));
  return (
    <div className="bars">
      {items.map((d, i) => (
        <div className="bar-row" key={i} title={`${d.label}: ${d.value}`}>
          <span className="bar-label" style={{ width: labelWidth }}>{d.label}</span>
          <span className="bar-track"><i style={{ width: `${(Math.abs(d.value) / m) * 100}%`, background: d.color || color }} /></span>
          <span className="bar-val">{format(d.value)}</span>
        </div>
      ))}
    </div>
  );
}

// 直方图 / histogram from {counts, min, max}
export function Hist({ hist, width = 120, height = 28, color = "#7aa2f7" }) {
  if (!hist?.counts) return null;
  const m = Math.max(1, ...hist.counts), n = hist.counts.length, bw = width / n;
  return (
    <svg width={width} height={height} className="hist">
      {hist.counts.map((c, i) => (
        <rect key={i} x={i * bw + 0.5} y={height - (c / m) * height} width={Math.max(1, bw - 1)} height={(c / m) * height} fill={color} />
      ))}
    </svg>
  );
}

// 散点图（canvas）：分类按颜色，回归按数值渐变；可选 y=x 参考线
// Scatter: categorical colors, or a value ramp for regression; optional y=x line.
export function Scatter({ points, mode = "class", size = 300, diag = false, xLabel, yLabel }) {
  const ref = useRef(null);
  const lang = useLang();
  useEffect(() => {
    const cv = ref.current, dpr = window.devicePixelRatio || 1, S = size;
    cv.width = S * dpr; cv.height = S * dpr;
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, S, S);
    if (!points?.length) return;
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (diag) { x0 = y0 = Math.min(x0, y0); x1 = y1 = Math.max(x1, y1); }
    const pad = 14, sx = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (S - 2 * pad), sy = (v) => S - pad - ((v - y0) / (y1 - y0 || 1)) * (S - 2 * pad);
    ctx.strokeStyle = "#262c3b"; ctx.strokeRect(0.5, 0.5, S - 1, S - 1);
    if (diag) { ctx.strokeStyle = "#7f879b"; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(sx(x0), sy(y0)); ctx.lineTo(sx(x1), sy(y1)); ctx.stroke(); ctx.setLineDash([]); }
    const vals = points.map((p) => p[2]);
    const v0 = Math.min(...vals), v1 = Math.max(...vals);
    for (const p of points) {
      ctx.fillStyle = mode === "class" ? classColor(p[2] ?? 0) : mode === "value" ? `hsl(${220 - 200 * ((p[2] - v0) / (v1 - v0 || 1))} 70% 60%)` : "#7aa2f7";
      ctx.globalAlpha = 0.8;
      ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 2.4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#7f879b"; ctx.font = "10px system-ui";
    if (xLabel) ctx.fillText(tr(xLabel), pad, S - 3);
    if (yLabel) { ctx.save(); ctx.translate(10, pad + 60); ctx.rotate(-Math.PI / 2); ctx.fillText(tr(yLabel), 0, 0); ctx.restore(); }
  }, [points, mode, size, diag, xLabel, yLabel, lang]);
  return <canvas ref={ref} style={{ width: size, height: size }} />;
}

// 混淆矩阵热力图 / confusion-matrix heat map
export function Confusion({ matrix, classes, cell = 26 }) {
  if (!matrix) return <div className="muted">类别太多，未显示混淆矩阵。</div>;
  const rowSum = matrix.map((r) => Math.max(1, r.reduce((a, b) => a + b, 0)));
  const short = (s) => (s.length > 6 ? s.slice(0, 6) + "…" : s);
  return (
    <div className="confusion">
      <div className="cm-axis muted">行 = 真实类别，列 = 预测类别（颜色 = 该行占比）</div>
      <table>
        <thead><tr><th />{classes.map((c, j) => <th key={j} title={c}>{short(c)}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th title={classes[i]}>{short(classes[i])}</th>
              {row.map((v, j) => {
                const a = v / rowSum[i];
                return <td key={j} style={{ width: cell, height: cell, background: i === j ? `rgba(158,206,106,${0.12 + a * 0.8})` : `rgba(247,118,142,${a * 0.9})` }}>{v || ""}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// 多条折线（SVG）：例如每类的平均频谱 / small multi-line chart
export function Lines({ series, labels, width = 520, height = 160 }) {
  const all = series.flatMap((s) => s.values);
  const lo = Math.min(...all), hi = Math.max(...all), n = series[0]?.values.length || 1;
  const X = (i) => 30 + (i / (n - 1 || 1)) * (width - 40), Y = (v) => 10 + (1 - (v - lo) / (hi - lo || 1)) * (height - 30);
  return (
    <svg width={width} height={height} className="lines">
      {series.map((s, k) => (
        <polyline key={k} fill="none" stroke={s.color} strokeWidth="1.8" points={s.values.map((v, i) => `${X(i)},${Y(v)}`).join(" ")} />
      ))}
      {labels && [0, Math.floor(n / 2), n - 1].map((i) => <text key={i} x={X(i)} y={height - 4} fill="#7f879b" fontSize="10" textAnchor="middle">{labels[i]}</text>)}
    </svg>
  );
}
