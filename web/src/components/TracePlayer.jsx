// 慢动作回放控制 + 时间轴（颜色 = 文件，高度 = 调用深度）
// Slow-motion playback controls + timeline strip (color = file, height = call depth).
import { useEffect, useRef } from "react";

export const FILE_COLORS = ["#7aa2f7", "#9ece6a", "#e0af68", "#bb9af7", "#f7768e", "#7dcfff", "#ff9e64"];

export default function TracePlayer({ trace, order, pos, setPos, playing, setPlaying, speed, setSpeed, quiet, setQuiet, onRecord, busy }) {
  const ref = useRef(null);
  const n = order.length;

  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const w = cv.clientWidth, h = 26, dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!trace || !n) return;
    const maxD = Math.max(...order.map((i) => trace.events[i].d || 1));
    for (let px = 0; px < w; px++) {
      const k = Math.floor((px / w) * n);
      const ev = trace.events[order[k]];
      ctx.fillStyle = ev.t === "note" ? "#f7768e" : FILE_COLORS[(ev.f ?? 0) % FILE_COLORS.length];
      const bh = 6 + (h - 8) * ((ev.d || 1) / maxD);
      ctx.fillRect(px, h - bh, 1, bh);
    }
    const cx = (pos / Math.max(1, n - 1)) * (w - 2);
    ctx.fillStyle = "#fff"; ctx.fillRect(cx, 0, 2, h);
  }, [trace, order, pos, n]);

  const seek = (e) => {
    const r = ref.current.getBoundingClientRect();
    setPos(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width) * (n - 1)))));
  };

  return (
    <div className="card player">
      <div className="player-row">
        <button className="btn primary" onClick={onRecord} disabled={busy} title="录制下一次迭代的逐行轨迹">● 单步讲解</button>
        <div className="sep" />
        <button className="btn icon" onClick={() => setPos(0)} disabled={!n} title="开头">⏮</button>
        <button className="btn icon" onClick={() => setPos(Math.max(0, pos - 1))} disabled={!n} title="上一行 (←)">◀</button>
        <button className="btn icon" onClick={() => setPlaying(!playing)} disabled={!n} title="播放/暂停 (空格)">{playing ? "⏸" : "▶"}</button>
        <button className="btn icon" onClick={() => setPos(Math.min(n - 1, pos + 1))} disabled={!n} title="下一行 (→)">▶|</button>
        <button className="btn icon" onClick={() => setPos(n - 1)} disabled={!n} title="结尾">⏭</button>
        <select value={speed} onChange={(e) => setSpeed(+e.target.value)} title="回放速度">
          {[0.25, 0.5, 1, 2, 4, 8].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={quiet} onChange={(e) => setQuiet(e.target.checked)} />只看产生数据的行</label>
        <span className="muted grow-right">{n ? `${pos + 1} / ${n}` : "尚未录制"}</span>
      </div>
      <canvas ref={ref} className="timeline" onClick={seek} style={{ width: "100%", height: 26 }} />
      {trace && (
        <div className="legend small">
          {trace.files.map((f, i) => <span key={f}><i style={{ background: FILE_COLORS[i % FILE_COLORS.length] }} />{f.split("/").pop()}</span>)}
          <span className="muted">· 第 {trace.step} 步 · loss {trace.loss?.toFixed?.(4)} · 录制用时 {trace.ms} ms{trace.truncated ? " · 已截断" : ""}</span>
        </div>
      )}
    </div>
  );
}
