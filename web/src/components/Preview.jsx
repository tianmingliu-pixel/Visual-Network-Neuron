// 预览：网络此刻学到了什么 / what the network has learned so far
import { useEffect, useRef } from "react";
import { fmt } from "../api.js";
import { Confusion, Scatter, Thumb } from "./viz.jsx";

function Pixels({ w, h, px, scale = 4, title }) {
  const ref = useRef(null);
  useEffect(() => {
    const ctx = ref.current.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const v = px[i];
      img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [w, h, px]);
  return <canvas ref={ref} width={w} height={h} title={title} className="px" style={{ width: w * scale, height: h * scale }} />;
}

const BCOL = [[247, 118, 142], [158, 206, 106], [122, 162, 247], [224, 175, 104]];

// 决策边界：每个格点按预测类别上色（越确定颜色越深），上面叠加验证集的真实点
function Boundary({ p }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current, S = 260, ctx = cv.getContext("2d");
    cv.width = S; cv.height = S;
    const n = p.n, cell = S / n;
    for (let i = 0; i < n * n; i++) {
      const c = BCOL[p.cls[i] % BCOL.length], a = 0.12 + 0.5 * Math.max(0, (p.conf[i] - 0.34) / 0.66);
      ctx.fillStyle = `rgba(${c},${a})`;
      ctx.fillRect((i % n) * cell, Math.floor(i / n) * cell, cell + 0.5, cell + 0.5);
    }
    const X = (v) => ((v + 1.1) / 2.2) * S, Y = (v) => ((1.1 - v) / 2.2) * S;
    for (const [x, y, k] of p.points) {
      ctx.beginPath(); ctx.arc(X(x), Y(y), 2.6, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${BCOL[k % BCOL.length]})`; ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = 0.8; ctx.stroke();
    }
  }, [p]);
  return (
    <div className="boundary">
      <canvas ref={ref} style={{ width: 260, height: 260 }} />
      <div className="b-side">
        <div className="big">{(p.val_acc * 100).toFixed(1)}%</div><div className="muted">验证集准确率</div>
        <p>背景颜色 = 网络对平面上每个位置的判断（越深越确定）；圆点 = 验证集的真实标签。训练中可以看到边界从直线慢慢弯成螺旋。</p>
        <div className="legend">{p.classes.map((c, i) => <span key={c}><i style={{ background: `rgb(${BCOL[i]})` }} />{c}</span>)}</div>
      </div>
    </div>
  );
}

export default function Preview({ preview }) {
  if (!preview) return <div className="card"><div className="card-head"><span>学习效果预览</span></div><div className="empty">开始训练后，这里每秒刷新一次网络的输出</div></div>;
  const { kind } = preview;
  return (
    <div className="card">
      <div className="card-head"><span>学习效果预览</span><span className="muted">step {preview.step}</span></div>
      {kind === "boundary" && <Boundary p={preview} />}
      {kind === "eval" && preview.task_type === "classification" && (
        <div className="boundary">
          <div className="b-side" style={{ flex: "0 0 auto", minWidth: 120 }}>
            <div className="big">{(preview.acc * 100).toFixed(1)}%</div><div className="muted">验证集准确率（{preview.n_val} 条）</div>
          </div>
          <Confusion matrix={preview.confusion} classes={preview.classes} cell={22} />
          {preview.items && (
            <div className="grid-imgs" style={{ flex: 1 }}>{preview.items.map((it, i) => (
              <figure key={i}><Thumb {...it} scale={56 / it.w} /><figcaption className={it.pred === it.label ? "ok" : "bad"}>
                {it.pred} <small>{Math.round(it.conf * 100)}%</small><br /><small className="muted">真实 {it.label}</small></figcaption></figure>))}</div>
          )}
        </div>
      )}
      {kind === "eval" && preview.task_type === "regression" && (
        <div className="boundary">
          <Scatter points={preview.points.map(([t, p]) => [t, p, 0])} mode="plain" size={240} diag xLabel={"真实 " + preview.target} yLabel="预测" />
          <div className="b-side">
            <div className="big">R² {fmt(preview.r2, 3)}</div><div className="muted">MAE {fmt(preview.mae)} · RMSE {fmt(preview.rmse)}</div>
            <p>每个点是一条验证样本：横轴真实值，纵轴预测值。越贴近虚线（预测 = 真实）越准。</p>
          </div>
        </div>
      )}
      {kind === "seq" && (
        <table className="seq">
          <thead><tr><th>输入</th><th>正确答案（倒序）</th><th>模型输出</th></tr></thead>
          <tbody>
            {preview.rows.map((r, i) => (
              <tr key={i}>
                <td>{r.src.join(" ")}</td>
                <td>{r.target.join(" ")}</td>
                <td>{r.target.map((t, j) => <span key={j} className={r.pred[j] === t ? "ok" : "bad"}>{r.pred[j] ?? "·"} </span>)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {kind === "classify" && (
        <div className="grid-imgs">
          {preview.items.map((it, i) => (
            <figure key={i}>
              <Pixels w={it.w} h={it.h} px={it.px} scale={2} />
              <figcaption className={it.pred === it.label ? "ok" : "bad"}>
                {it.pred} <small>{Math.round(it.conf * 100)}%</small><br /><small className="muted">真实: {it.label}</small>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
      {kind === "denoise" && (
        <div className="denoise">
          <div className="dn-head"><span>噪声步 t</span><span>原图 x₀</span><span>加噪 xₜ</span><span>网络还原 x̂₀</span></div>
          {preview.rows.map((r, i) => (
            <div className="dn-row" key={i}>
              <span className="muted">t={r.t}</span>
              <Pixels w={r.w} h={r.h} px={r.clean} />
              <Pixels w={r.w} h={r.h} px={r.noisy} />
              <Pixels w={r.w} h={r.h} px={r.pred} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
