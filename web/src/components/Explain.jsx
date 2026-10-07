// 讲解面板：这一行在做什么 + 它产生了哪些张量（形状 / 统计 / 分布条）
// Explain panel: what the current line does and which tensors it produced.
import { fmt } from "../api.js";

function Dist({ v }) {
  if (v.min == null || v.max == null) return null;
  const lo = v.min, hi = v.max, span = hi - lo || 1;
  const x = (u) => ((u - lo) / span) * 100;
  const m = v.mean ?? 0, s = v.std ?? 0;
  return (
    <svg className="dist" viewBox="0 0 100 10" preserveAspectRatio="none">
      <rect x="0" y="4" width="100" height="2" className="dist-range" />
      <rect x={Math.max(0, x(m - s))} y="2" width={Math.max(1, Math.min(100, x(m + s)) - Math.max(0, x(m - s)))} height="6" className="dist-std" />
      <rect x={Math.min(99, Math.max(0, x(m)))} y="0" width="1.2" height="10" className="dist-mean" />
    </svg>
  );
}

function VarRow({ v }) {
  if (v.k === "tensor") {
    return (
      <tr>
        <td className="vname">{v.n}</td>
        <td><code>[{v.s.join(", ")}]</code> <span className="muted">{v.dt}</span>{v.grad ? <span className="badge">grad</span> : null}</td>
        <td>{fmt(v.mean, 3)} ± {fmt(v.std, 3)}</td>
        <td className="muted">[{fmt(v.min, 2)}, {fmt(v.max, 2)}]</td>
        <td style={{ width: 90 }}><Dist v={v} /></td>
      </tr>
    );
  }
  if (v.k === "list") {
    return <tr><td className="vname">{v.n}</td><td colSpan={4}>{v.items.map((it, i) => <code key={i}>[{it.s?.join(", ")}] </code>)}</td></tr>;
  }
  return <tr><td className="vname">{v.n}</td><td colSpan={4}><code>{JSON.stringify(v.val)}</code></td></tr>;
}

export default function Explain({ trace, ev, path, source }) {
  if (!trace || !ev) {
    return (
      <div className="card explain">
        <div className="card-head"><span>这一行在做什么</span></div>
        <div className="empty">
          点击 <b>● 单步讲解</b>：后台会把下一次训练迭代（前向传播 → 损失 → 反向传播 → 参数更新）逐行录下来，
          然后在这里像放慢动作一样回放：左边代码高亮当前行，这里显示中文注释和该行产生的张量。
          <br /><br />平时的全速训练不追踪，所以不会变慢。
        </div>
      </div>
    );
  }
  if (ev.t === "note") {
    return (
      <div className="card explain">
        <div className="card-head"><span>⚙ {ev.title}</span></div>
        <p className="say">{ev.detail}</p>
        <div className="vars"><table><tbody>{ev.v.map((v, i) => <VarRow key={i} v={v} />)}</tbody></table></div>
      </div>
    );
  }
  const line = source?.lines?.[ev.l - 1];
  const kind = ev.t === "call" ? "进入函数" : "执行";
  return (
    <div className="card explain">
      <div className="card-head">
        <span className="crumbs">{path.map((p, i) => <span key={i}>{i ? " › " : ""}{p}</span>)}</span>
        <span className="muted">{trace.files[ev.f]}:{ev.l}</span>
      </div>
      <div className="now">
        <span className={"kind " + ev.t}>{kind}</span>
        <code>{line ? line.code.trim() : ""}</code>
      </div>
      <p className="say">{line?.comment || (ev.t === "call" ? `调用 ${ev.c ? ev.c + "." : ""}${ev.fn}()` : "（此行无注释）")}</p>
      {ev.v?.length ? (
        <div className="vars">
          <div className="vars-title">{ev.t === "call" ? "传入参数" : "本行产生 / 更新的变量"}</div>
          <table>
            <thead><tr><th>变量</th><th>形状 · 类型</th><th>均值 ± 标准差</th><th>范围</th><th>分布</th></tr></thead>
            <tbody>{ev.v.map((v, i) => <VarRow key={i} v={v} />)}</tbody>
          </table>
        </div>
      ) : null}
      {ev.ret ? (
        <div className="vars">
          <div className="vars-title">返回值</div>
          <table><tbody><VarRow v={{ ...ev.ret, n: "return" }} /></tbody></table>
        </div>
      ) : null}
    </div>
  );
}
