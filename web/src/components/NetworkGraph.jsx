// 网络结构图 / Network diagram
//  · MLP：画出每个神经元和每条权重（橙=正权重，蓝=负权重，粗细=|w|），节点颜色=当前样本的激活值，
//          前向脉冲（青色）从输入流向输出，反向脉冲（品红）沿梯度大的连线流回；右侧是输出概率条。
//  · 大网络（ViT/Transformer/U-Net/DiT）：每一列是一层（按真实执行顺序），列里的点是抽样的代表性神经元，
//          顶部品红短条 = 该层梯度范数；悬停看形状/参数/统计，点击跳到该层的代码。
// 静态部分画在离屏画布上，只在数据更新时重画；动画每帧只叠加脉冲，保证流畅。
import { useEffect, useMemo, useRef, useState } from "react";
import { fmt } from "../api.js";
import { tr } from "../i18n/index.js";

const POS = [224, 140, 70], NEG = [79, 159, 230], FWD = "#5ee0ff", BWD = "#ff5fd2";
const CLASS_COLORS = ["#f7768e", "#9ece6a", "#7aa2f7", "#e0af68", "#bb9af7"];

function divColor(v, alpha = 1) {            // v ∈ [-1, 1] → 蓝(负) … 灰 … 橙(正)
  const t = Math.max(-1, Math.min(1, v || 0));
  const base = [58, 64, 82], tgt = t >= 0 ? POS : NEG, k = Math.abs(t);
  const c = base.map((b, i) => Math.round(b + (tgt[i] - b) * k));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

function useSize(ref) {
  const [w, setW] = useState(600);
  useEffect(() => {
    const ro = new ResizeObserver((es) => setW(Math.max(320, es[0].contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

// ---- 布局：MLP 每个神经元的位置 / MLP layout ------------------------------------------
function mlpLayout(g, W, H) {
  const L = g.sizes.length, left = 92, right = W - 170, top = 26, bottom = H - 34;
  return g.sizes.map((n, l) => {
    const x = left + ((right - left) * l) / (L - 1);
    const gap = Math.min(22, (bottom - top) / Math.max(1, n - 1));
    const y0 = (top + bottom) / 2 - (gap * (n - 1)) / 2;
    return Array.from({ length: n }, (_, i) => ({ x, y: y0 + i * gap }));
  });
}

function drawMLP(ctx, g, W, H, pos) {
  ctx.clearRect(0, 0, W, H);
  const maxW = Math.max(1e-6, ...g.W.flat(2).map(Math.abs));
  // 连线：颜色=符号，透明度/粗细=|w| / edges
  g.W.forEach((M, l) => {
    M.forEach((row, j) => row.forEach((w, i) => {
      const a = pos[l][i], b = pos[l + 1][j], k = Math.abs(w) / maxW;
      ctx.strokeStyle = `rgba(${(w >= 0 ? POS : NEG).join(",")},${0.06 + 0.7 * k})`;
      ctx.lineWidth = 0.4 + 2.2 * k;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }));
  });
  // 神经元：颜色=激活值 / nodes colored by activation
  const L = g.sizes.length;
  g.acts.forEach((acts, l) => {
    const scale = l === 0 ? Math.max(1e-6, ...acts.map(Math.abs)) : 1;
    acts.forEach((v, i) => {
      const p = pos[l][i];
      const val = l === L - 1 ? (g.regression ? Math.tanh(v) : g.probs[i] * 2 - 1) : v / scale;
      ctx.beginPath(); ctx.arc(p.x, p.y, l === L - 1 ? 8 : 6.5, 0, Math.PI * 2);
      ctx.fillStyle = divColor(val); ctx.fill();
      ctx.lineWidth = 1.2; ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.stroke();
    });
  });
  // 输入特征名与数值 / input labels
  ctx.font = "11px system-ui"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  g.in_names.forEach((nm, i) => {
    ctx.fillStyle = "#aab2c5"; ctx.fillText(`${nm}  ${(g.acts[0][i] ?? 0).toFixed(2)}`, pos[0][i].x - 12, pos[0][i].y);
  });
  // 输出：类别名 + 概率条 / output bars
  ctx.textAlign = "left";
  const last = pos[L - 1];
  if (g.regression) {                                   // 回归：显示预测值与真实值 / regression output
    const p = last[0];
    ctx.fillStyle = "#e0af68"; ctx.fillText(tr(g.out_names[0]), p.x + 14, p.y - 9);
    ctx.fillStyle = "#d7dce8"; ctx.fillText(tr(`预测 ${(+g.regression.pred).toPrecision(4)}`), p.x + 14, p.y + 6);
    ctx.fillStyle = "#7f879b"; ctx.fillText(tr(`真实 ${(+g.regression.target).toPrecision(4)}`), p.x + 14, p.y + 20);
  } else g.out_names.forEach((nm, i) => {
    const p = last[i], pr = g.probs[i];
    ctx.fillStyle = CLASS_COLORS[i % CLASS_COLORS.length]; ctx.fillText(tr(nm), p.x + 14, p.y);
    const bx = p.x + 52, bw = W - bx - 44;
    ctx.fillStyle = "rgba(255,255,255,.07)"; ctx.fillRect(bx, p.y - 5, bw, 10);
    ctx.fillStyle = CLASS_COLORS[i % CLASS_COLORS.length]; ctx.fillRect(bx, p.y - 5, bw * pr, 10);
    if (i === g.label) { ctx.fillStyle = "#fff"; ctx.fillRect(bx + bw - 2, p.y - 7, 2, 14); }
    ctx.fillStyle = "#d7dce8"; ctx.fillText((pr * 100).toFixed(0) + "%", bx + bw + 6, p.y);
  });
  // 层名 / layer titles
  ctx.textAlign = "center"; ctx.fillStyle = "#7f879b"; ctx.font = "12px system-ui";
  const names = ["输入层", ...g.sizes.slice(1, -1).map((_, i) => `隐藏层 ${i + 1}`), "输出层"].map(tr);
  const ts = g.true_sizes || g.sizes;
  pos.forEach((col, l) => ctx.fillText(`${names[l]} (${ts[l] > g.sizes[l] ? g.sizes[l] + "/" + ts[l] : ts[l]})`, col[0].x, H - 12));
}

// ---- 大网络：层流程图 / architecture flow ------------------------------------------------
function archLayout(g, W, H) {
  const cols = [{ name: "input", cls: "输入", ...g.input }, ...g.nodes];
  const left = 30, right = W - 30, top = 40, bottom = H - 70;
  return cols.map((c, k) => {
    const x = left + ((right - left) * k) / Math.max(1, cols.length - 1);
    const n = (c.units || []).length || 1;
    const gap = Math.min(18, (bottom - top) / Math.max(1, n - 1));
    const y0 = (top + bottom) / 2 - (gap * (n - 1)) / 2;
    return { col: c, x, pts: Array.from({ length: n }, (_, i) => ({ x, y: y0 + i * gap })) };
  });
}

function drawArch(ctx, g, W, H, lay, hover) {
  ctx.clearRect(0, 0, W, H);
  const maxG = Math.max(1e-9, ...g.nodes.map((n) => n.gnorm || 0));
  for (let k = 0; k + 1 < lay.length; k++) {           // 相邻层全连接的示意线 / symbolic edges
    const A = lay[k], B = lay[k + 1];
    ctx.strokeStyle = "rgba(122,162,247,.08)"; ctx.lineWidth = 0.6;
    for (const a of A.pts) for (const b of B.pts) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
  }
  lay.forEach((L, k) => {
    const u = L.col.units || [];
    const m = Math.max(1e-6, ...u.map((v) => Math.abs(v ?? 0)));
    if (hover === k) { ctx.fillStyle = "rgba(122,162,247,.12)"; ctx.fillRect(L.x - 13, 20, 26, H - 80); }
    L.pts.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = divColor((u[i] ?? 0) / m); ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,.3)"; ctx.lineWidth = 1; ctx.stroke();
    });
    if (k > 0 && L.col.gnorm) {                          // 梯度范数条 / gradient-norm bar
      const h = 4 + 14 * Math.sqrt(L.col.gnorm / maxG);
      ctx.fillStyle = BWD; ctx.fillRect(L.x - 3, 32 - h, 6, h);
    }
    ctx.save(); ctx.translate(L.x, H - 62); ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = hover === k ? "#fff" : "#8a93a8"; ctx.font = "10.5px system-ui"; ctx.textAlign = "right";
    ctx.fillText(k === 0 ? tr("输入") : L.col.name, 0, 0); ctx.restore();
  });
}

// ---- 组件 / component ---------------------------------------------------------------------
export default function NetworkGraph({ graph, phase, onJump, taskTitle, height, lang }) {
  const wrapRef = useRef(null), cvRef = useRef(null), staticRef = useRef(null);
  const W = useSize(wrapRef);
  const [hover, setHover] = useState(null);
  const isMLP = graph?.kind === "mlp";
  const H = height ?? (isMLP ? 380 : 330);
  const minW = graph && !isMLP ? Math.max(W, 60 + (graph.nodes.length + 1) * 34) : W;

  const layout = useMemo(() => {
    if (!graph) return null;
    return isMLP ? mlpLayout(graph, minW, H) : archLayout(graph, minW, H);
  }, [graph, minW, H, isMLP, lang]);

  // 静态层：数据变化时重画到离屏画布 / static layer
  useEffect(() => {
    if (!graph || !layout) return;
    const dpr = window.devicePixelRatio || 1;
    const off = staticRef.current || (staticRef.current = document.createElement("canvas"));
    off.width = minW * dpr; off.height = H * dpr;
    const ctx = off.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (isMLP) drawMLP(ctx, graph, minW, H, layout); else drawArch(ctx, graph, minW, H, layout, hover);
  }, [graph, layout, minW, H, isMLP, hover]);

  // 动画层：脉冲 / animated pulses
  const phaseRef = useRef(phase); phaseRef.current = phase;
  const t0Ref = useRef(performance.now());
  useEffect(() => {
    if (!graph || !layout) return;
    const cv = cvRef.current, dpr = window.devicePixelRatio || 1;
    cv.width = minW * dpr; cv.height = H * dpr;
    const ctx = cv.getContext("2d");
    // 预先挑出脉冲走的连线：前向=|w|最大，反向=|∇w|最大 / pick edges for pulses
    const segs = [];
    if (isMLP) {
      graph.W.forEach((M, l) => {
        const all = [];
        M.forEach((row, j) => row.forEach((w, i) => all.push({ a: layout[l][i], b: layout[l + 1][j], w: Math.abs(w), g: Math.abs(graph.G?.[l]?.[j]?.[i] ?? 0) })));
        segs.push({ f: [...all].sort((x, y) => y.w - x.w).slice(0, 14), b: [...all].sort((x, y) => y.g - x.g).slice(0, 14) });
      });
    } else {
      for (let k = 0; k + 1 < layout.length; k++) {
        const A = layout[k].pts, B = layout[k + 1].pts, s = [];
        for (let i = 0; i < Math.max(A.length, B.length); i++) s.push({ a: A[i % A.length], b: B[(i * 3) % B.length] });
        segs.push({ f: s, b: s });
      }
    }
    let raf;
    const t0 = t0Ref.current;                      // 跨数据更新保持动画相位 / keep phase across updates
    const loop = (now) => {
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.drawImage(staticRef.current, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const ph = phaseRef.current;
      const back = ph === "backward" || ph === "update";
      if (ph !== "idle" && segs.length) {
        const T = 1800, p = ((now - t0) % T) / T * segs.length;
        const s = Math.floor(p), frac = p - s;
        const seg = back ? segs[segs.length - 1 - s] : segs[s];
        const list = back ? seg.b : seg.f;
        ctx.fillStyle = back ? BWD : FWD;
        ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 8;
        for (const e of list) {
          const [a, b] = back ? [e.b, e.a] : [e.a, e.b];
          ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac, 2.6, 0, Math.PI * 2); ctx.fill();
        }
        ctx.shadowBlur = 0;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [graph, layout, minW, H, isMLP, lang]);

  // 悬停与点击 / hover & click
  const hitColumn = (e) => {
    if (!layout) return null;
    const r = cvRef.current.getBoundingClientRect(), x = e.clientX - r.left;
    let best = null, bd = 1e9;
    (isMLP ? layout.map((col) => col[0].x) : layout.map((L) => L.x)).forEach((cx, k) => {
      const d = Math.abs(cx - x); if (d < bd) { bd = d; best = k; }
    });
    return bd < 26 ? best : null;
  };
  const colInfo = (k) => {
    if (k == null || !graph) return null;
    if (isMLP) {
      const L = graph.sizes.length, refs = graph.refs || {};
      if (k === 0) return { title: "输入层", lines: [`${(graph.true_sizes || graph.sizes)[0]} 个特征（画出 ${graph.sizes[0]} 个）：${graph.in_names.join("、")}`], ref: refs.input };
      if (k === L - 1) return graph.regression
        ? { title: "输出层", lines: ["z = W·a + b（回归：直接输出一个数）", `预测 ${graph.regression.pred}　真实 ${graph.regression.target}`], ref: refs.output }
        : { title: "输出层", lines: ["z = W·a + b → softmax → 概率", `预测：${graph.out_names[graph.probs.indexOf(Math.max(...graph.probs))]}　真实：${graph.out_names[graph.label]}`], ref: refs.output };
      const Wl = graph.W[k - 1], Gl = graph.G?.[k - 1];
      const wn = Math.sqrt(Wl.flat().reduce((s, v) => s + v * v, 0)), gn = Gl ? Math.sqrt(Gl.flat().reduce((s, v) => s + v * v, 0)) : null;
      return { title: `隐藏层 ${k}`, lines: [`${(graph.true_sizes || graph.sizes)[k]} 个神经元，a = tanh(W·a_prev + b)`, `W 形状 ${Wl.length}×${Wl[0].length}，‖W‖=${fmt(wn, 3)}，‖∇W‖=${fmt(gn, 4)}`], ref: refs.linear };
    }
    const c = k === 0 ? { cls: "输入", ...graph.input, name: "input" } : graph.nodes[k - 1];
    return {
      title: `${c.name} · ${c.cls}`,
      lines: [`输出形状 [${(c.shape || []).join(", ")}]`, `均值 ${fmt(c.mean, 3)} ± ${fmt(c.std, 3)}`,
        k ? `参数 ${(c.params || 0).toLocaleString()}　‖W‖=${fmt(c.wnorm, 3)}　‖∇W‖=${fmt(c.gnorm, 4)}` : `类型 ${c.dtype}`],
      ref: c.file ? { file: c.file, line: c.line } : (k === 0 ? null : { file: graph.file, line: graph.line }),
    };
  };
  const info = colInfo(hover);

  if (!graph) {
    return <div className="card" ref={wrapRef}><div className="card-head"><span>神经网络 · 结构图</span></div>
      <div className="empty">开始训练后显示网络结构：输入层 → 隐藏层 → 输出层，以及权重、激活和梯度。</div></div>;
  }
  const pred = isMLP ? graph.probs.indexOf(Math.max(...graph.probs)) : null;
  return (
    <div className="card netgraph" ref={wrapRef}>
      <div className="card-head">
        <span>神经网络 · 结构图 <small className="muted">{taskTitle}</small></span>
        <span className="muted small-txt">step {graph.step}</span>
      </div>
      <div className="ng-sub">
        {isMLP ? (
          <>
            <b>MLP {(graph.true_sizes || graph.sizes).join(" → ")}</b>
            {graph.regression ? <span>验证样本 #{graph.sample_index}　预测 {(+graph.regression.pred).toPrecision(4)}　真实 {(+graph.regression.target).toPrecision(4)}</span> : (
            <span>{graph.sample_index != null ? `验证样本 #${graph.sample_index}` : `样本 (${graph.point.map((v) => v.toFixed(2)).join(", ")})`}　真实 <i style={{ color: CLASS_COLORS[graph.label % 5] }}>{graph.out_names[graph.label]}</i>
              　预测 <i style={{ color: CLASS_COLORS[pred % 5] }}>{graph.out_names[pred]}</i> {(graph.probs[pred] * 100).toFixed(0)}%</span>)}
            {graph.true_sizes && graph.true_sizes.some((n, i) => n > graph.sizes[i]) && <span className="muted">（神经元较多，只画出权重最大的那些）</span>}
          </>
        ) : (
          <>
            <b>{graph.model} · {graph.nodes.length} 层 · {graph.total_params.toLocaleString()} 参数</b>
            <span>输入 [{(graph.input.shape || []).join(", ")}] → 输出 [{(graph.nodes.at(-1)?.shape || []).join(", ")}]　批损失 {fmt(graph.loss)}</span>
          </>
        )}
        <span className="legend">
          {isMLP && <><span><i style={{ background: `rgb(${POS})` }} />正权重</span><span><i style={{ background: `rgb(${NEG})` }} />负权重</span></>}
          <span><i style={{ background: FWD }} />前向传播</span><span><i style={{ background: BWD }} />反向传播{isMLP ? "" : "（顶部条=梯度范数）"}</span>
        </span>
      </div>
      <div className="ng-scroll">
        <canvas ref={cvRef} style={{ width: minW, height: H, cursor: hover != null ? "pointer" : "default" }}
          onMouseMove={(e) => setHover(hitColumn(e))} onMouseLeave={() => setHover(null)}
          onClick={(e) => { const i = colInfo(hitColumn(e)); if (i?.ref) onJump(i.ref); }} />
      </div>
      <div className="ng-info">
        {info ? (<><b>{info.title}</b>{info.lines.map((l, i) => <span key={i}>{l}</span>)}{info.ref && <em>点击跳到代码 {info.ref.file}:{info.ref.line}</em>}</>)
          : <span className="muted">把鼠标移到某一层上查看细节，点击跳转到对应代码。节点颜色：橙=正值，蓝=负值，越亮绝对值越大。</span>}
      </div>
    </div>
  );
}
