// 机器记忆库面板 / Neural memory store panel
// 训练时每隔 N 步把“机器此刻的状态”写进本地 SQLite：指标、每层权重变化、每个样本的特征向量 + 元数据、检查点。
// 下次训练可以：从某个快照继续（权重 + 优化器动量），难例回放（多练以前错过的样本），图片增强。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch, fmt, getJSON, readJSON } from "../api.js";
import { classColor } from "./viz.jsx";
import { tr } from "../i18n/index.js";

const post = async (url, body) => readJSON(await apiFetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const bytes = (n) => (n > 1048576 ? (n / 1048576).toFixed(1) + " MB" : (n / 1024).toFixed(0) + " KB");
const TABS = [
  ["growth", "成长曲线"], ["space", "特征向量空间"], ["layers", "每层变化"], ["samples", "样本记忆"], ["schema", "数据结构 · SQL"],
];
const PRESETS = [
  ["每次运行的最佳成绩", "SELECT id, title, steps, ROUND(best_val,4) AS best_val, parent_snapshot FROM runs ORDER BY created DESC"],
  ["最难的样本", "SELECT s.source, s.label, m.seen, m.wrong, ROUND(m.ema_loss,4) AS ema_loss, m.last_pred FROM sample_memory m JOIN samples s USING(dataset, sample_id) ORDER BY 1.0*m.wrong/m.seen DESC, m.ema_loss DESC LIMIT 30"],
  ["变化最大的层", "SELECT layer, ROUND(AVG(delta),5) AS avg_delta, ROUND(AVG(gnorm),5) AS avg_grad FROM layer_states GROUP BY layer ORDER BY avg_delta DESC"],
  ["特征向量记录", "SELECT snapshot_id, sample_id, split, dim, vector, label, pred, ROUND(confidence,3) AS conf FROM embeddings ORDER BY snapshot_id DESC LIMIT 20"],
];

export default function MemoryPanel({ memOpts, setMemOpts, task, isCustom, memTick, running, log }) {
  const [ov, setOv] = useState(null);
  const [runId, setRunId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [snapId, setSnapId] = useState(null);
  const [snap, setSnap] = useState(null);
  const [tab, setTab] = useState("growth");
  const [err, setErr] = useState("");
  const follow = useRef(true);                // 跟随正在训练的那次运行 / follow the live run

  const loadOv = useCallback(() => getJSON("/api/memory/overview").then((o) => {
    setOv(o); setErr("");
    const cur = o.current?.run_id;
    setRunId((r) => (follow.current && cur ? cur : r || cur || o.runs[0]?.id || null));
  }).catch((e) => setErr(e.message)), []);
  useEffect(() => { loadOv(); }, [loadOv, memTick]);
  useEffect(() => {
    if (!runId) { setDetail(null); return; }
    getJSON("/api/memory/run?id=" + encodeURIComponent(runId)).then((d) => {
      setDetail(d);
      setSnapId((s) => (follow.current || !d.snapshots.some((x) => x.id === s) ? d.snapshots[d.snapshots.length - 1]?.id ?? null : s));
    }).catch(() => setDetail(null));
  }, [runId, memTick]);
  useEffect(() => {
    if (!snapId) { setSnap(null); return; }
    getJSON("/api/memory/snapshot?id=" + snapId).then(setSnap).catch(() => setSnap(null));
  }, [snapId]);

  const run = detail?.run;
  const pickRun = (id) => { follow.current = id === ov?.current?.run_id; setRunId(id); };
  const resumeFrom = (s) => {
    setMemOpts({ ...memOpts, resume: s.id, memory: true });
    log?.(`下次「开始训练」将从记忆快照 #${s.id}（第 ${s.step} 步，验证 ${pct(s.val_acc)}）继续`);
  };
  const delRun = async (id) => {
    if (!confirm(tr("删除这次运行的全部记忆（快照、特征向量、权重文件）？"))) return;
    try { await post("/api/memory/action", { action: "delete_run", run_id: id }); setRunId(null); loadOv(); } catch (e) { setErr(e.message); }
  };

  return (
    <div className="card mem">
      <div className="card-head">
        <span>🧠 机器记忆库 <small className="muted">SQLite · 向量 + 元数据</small></span>
        {ov && <span className="mem-counts">
          {[["runs", "次训练"], ["snapshots", "个快照"], ["embeddings", "条特征向量"], ["sample_memory", "个样本记忆"]].map(([k, t]) =>
            <span key={k} className="chip"><b>{ov.counts[k]}</b> {t}</span>)}
          <span className="muted small-txt" title={ov.path}>{bytes(ov.bytes)}</span>
        </span>}
      </div>

      <Options memOpts={memOpts} setMemOpts={setMemOpts} isCustom={isCustom} task={task} running={running} ov={ov} />

      <div className="mem-tabs">
        {TABS.map(([k, t]) => <button key={k} className={"mtab" + (tab === k ? " on" : "")} onClick={() => setTab(k)}>{t}</button>)}
        <select className="run-pick" value={runId || ""} onChange={(e) => pickRun(e.target.value)}>
          {!ov?.runs.length && <option value="">还没有记忆——开始一次训练</option>}
          {ov?.runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id === ov.current?.run_id ? "● " : ""}{r.title} · {r.steps} 步 · 最佳 {pct(r.best_val)}{r.parent_snapshot ? " · 续自#" + r.parent_snapshot : ""}
            </option>))}
        </select>
        {run && <button className="btn small" onClick={() => delRun(run.id)} title="删除这次运行的记忆">🗑</button>}
      </div>
      {err && <div className="err">⚠ {err}</div>}

      <div className="mem-body">
        {tab === "growth" && <Growth detail={detail} ov={ov} snapId={snapId} setSnapId={(id) => { follow.current = false; setSnapId(id); }} resumeFrom={resumeFrom} />}
        {tab === "space" && <Space detail={detail} snap={snap} snapId={snapId} setSnapId={(id) => { follow.current = false; setSnapId(id); }} />}
        {tab === "layers" && <Layers detail={detail} />}
        {tab === "samples" && <Samples dataset={run?.dataset} tick={memTick} />}
        {tab === "schema" && <Schema ov={ov} />}
      </div>
    </div>
  );
}

// ---- 训练选项：下次训练如何使用记忆 / how the next run uses memory ---------------------------
function Options({ memOpts, setMemOpts, isCustom, task, running, ov }) {
  const [cands, setCands] = useState([]);
  useEffect(() => {
    const ds = ov?.current?.task === task ? ov?.current?.dataset : "";
    getJSON(`/api/memory/resumable?task=${task}${ds ? "&dataset=" + encodeURIComponent(ds) : ""}`).then((r) => setCands(r.items)).catch(() => setCands([]));
  }, [task, ov]);
  const set = (k, v) => setMemOpts({ ...memOpts, [k]: v });
  return (
    <div className="mem-opts">
      <label title="训练时每隔一段步数存一个快照"><input type="checkbox" checked={memOpts.memory} onChange={(e) => set("memory", e.target.checked)} />写入记忆</label>
      <label title="下次训练从哪里开始">起点
        <select value={memOpts.resume || ""} onChange={(e) => set("resume", e.target.value || null)} disabled={running}>
          <option value="">从零开始</option>
          <option value="best">从最佳记忆继续</option>
          {cands.map((c) => <option key={c.id} value={c.id}>快照 #{c.id} · 第 {c.step} 步 · 验证 {pct(c.val_acc)}{c.is_best ? " ★" : ""}</option>)}
        </select>
      </label>
      <label className={isCustom ? "" : "dis"} title="按记忆里的错误次数和损失给训练样本加权：以前错过的样本多练"><input type="checkbox" disabled={!isCustom} checked={memOpts.replay} onChange={(e) => set("replay", e.target.checked)} />难例回放</label>
      <label className={isCustom ? "" : "dis"} title="图片任务：随机左右翻转 + 平移，减少死记硬背"><input type="checkbox" disabled={!isCustom} checked={memOpts.augment} onChange={(e) => set("augment", e.target.checked)} />图片增强</label>
      <label title="每隔多少步存一个快照（0 = 自动：本次步数的 1/40）">快照间隔<input type="number" min="0" value={memOpts.mem_every} onChange={(e) => set("mem_every", +e.target.value)} /></label>
      {!isCustom && <span className="muted small-txt">演示任务只记录指标/层变化/权重；导入自己的数据后还会记录每个样本的特征向量。</span>}
    </div>
  );
}

// ---- 成长曲线：每个快照的训练/验证成绩，加上这次记忆的来源链 / growth across snapshots and runs -----
function Growth({ detail, ov, snapId, setSnapId, resumeFrom }) {
  if (!detail) return <Empty />;
  const S = detail.snapshots;
  const sel = S.find((s) => s.id === snapId);
  const related = (ov?.runs || []).filter((r) => r.dataset && r.dataset === detail.run.dataset && r.task === detail.run.task);
  return (
    <div className="growth">
      <SnapChart snaps={S} snapId={snapId} onPick={setSnapId} />
      <div className="growth-side">
        {detail.lineage.length > 0 && (
          <div className="lineage"><b>记忆来源</b>
            {[...detail.lineage].reverse().map((l) => <span key={l.id} className="chip">#{l.id} · {l.step} 步 · {pct(l.val_acc)}</span>)}
            <span className="chip on">本次</span>
          </div>
        )}
        {related.length > 1 && (
          <div className="runs-mini"><b>同一份数据的历次训练</b>
            {related.slice(0, 8).map((r) => (
              <div key={r.id} className={"rm" + (r.id === detail.run.id ? " on" : "")}>
                <span className="muted">{new Date(r.created * 1000).toLocaleTimeString()}</span>
                <span className="rm-bar"><i style={{ width: `${(r.best_val || 0) * 100}%` }} /></span>
                <span>{pct(r.best_val)}</span>
              </div>))}
          </div>
        )}
        {sel && (
          <div className="snap-card">
            <b>快照 #{sel.id}</b> 第 {sel.step} 步 {sel.is_best ? <span className="best">★ 最佳</span> : null}
            <div className="kv">
              <span>训练准确率</span><b>{pct(sel.train_acc)}</b><span>验证准确率</span><b>{pct(sel.val_acc)}</b>
              <span>loss</span><b>{fmt(sel.loss)}</b><span>验证 loss</span><b>{fmt(sel.val_loss)}</b>
              <span>权重变化</span><b>{fmt(sel.delta_norm)}</b><span>学习率</span><b>{fmt(sel.lr)}</b>
            </div>
            {sel.ckpt ? <button className="btn small primary" onClick={() => resumeFrom(sel)}>下次从这里继续训练</button>
              : <span className="muted small-txt">只保留每次运行“最新”和“最佳”的权重文件</span>}
          </div>
        )}
      </div>
    </div>
  );
}

function SnapChart({ snaps, snapId, onPick, width = 420, height = 180 }) {
  if (!snaps.length) return <Empty msg="这次运行还没有快照（训练到第一个快照间隔后出现）。" />;
  const x0 = snaps[0].step, x1 = snaps[snaps.length - 1].step;
  const X = (s) => 34 + ((s - x0) / (x1 - x0 || 1)) * (width - 44);
  const Y = (v) => 24 + (1 - Math.max(0, Math.min(1, v))) * (height - 44);
  const lmax = Math.max(1e-9, ...snaps.map((s) => s.loss || 0));
  const line = (key, scale = (v) => v) => snaps.filter((s) => s[key] != null).map((s) => `${X(s.step)},${Y(scale(s[key]))}`).join(" ");
  return (
    <svg width={width} height={height} className="snapchart">
      {[0, 0.5, 1].map((v) => <g key={v}><line x1="34" x2={width - 10} y1={Y(v)} y2={Y(v)} stroke="#232838" /><text x="2" y={Y(v) + 4} fill="#7f879b" fontSize="10">{v * 100}%</text></g>)}
      <polyline points={line("loss", (v) => v / lmax)} fill="none" stroke="#7aa2f7" strokeOpacity=".5" strokeWidth="1.2" />
      <polyline points={line("train_acc")} fill="none" stroke="#9ece6a" strokeWidth="1.6" />
      <polyline points={line("val_acc")} fill="none" stroke="#e0af68" strokeWidth="2.2" />
      {snaps.map((s) => (
        <g key={s.id} onClick={() => onPick(s.id)} style={{ cursor: "pointer" }}>
          <circle cx={X(s.step)} cy={Y(s.val_acc ?? s.train_acc ?? 0)} r={s.id === snapId ? 5 : s.is_best ? 4 : 2.6}
            fill={s.is_best ? "#e0af68" : "#c0caf5"} stroke={s.id === snapId ? "#fff" : "none"} />
          <title>{`#${s.id} 第${s.step}步  训练${pct(s.train_acc)}  验证${pct(s.val_acc)}`}</title>
        </g>))}
      <text x="34" y={height - 4} fill="#7f879b" fontSize="10">第 {x0} 步</text>
      <text x={width - 10} y={height - 4} fill="#7f879b" fontSize="10" textAnchor="end">第 {x1} 步</text>
      <g fontSize="10"><text x={width - 190} y="11" fill="#9ece6a">— 训练准确率</text><text x={width - 120} y="11" fill="#e0af68">— 验证准确率</text><text x={width - 50} y="11" fill="#7aa2f7">— loss</text></g>
    </svg>
  );
}

// ---- 特征向量空间：每个样本在网络“眼中”的位置，随快照变化 / embedding space per snapshot ---------
function Space({ detail, snap, snapId, setSnapId }) {
  const S = detail?.snapshots?.filter((s) => s.n_emb > 0) || [];
  const [rec, setRec] = useState(0);
  if (!detail) return <Empty />;
  if (!S.length) return <Empty msg="这次运行没有特征向量：只有导入自己的数据（自定义任务）时才记录每个样本的向量。" />;
  const i = Math.max(0, S.findIndex((s) => s.id === snapId));
  const P = snap?.projection;
  return (
    <div className="space">
      <div className="space-ctl">
        <input type="range" min="0" max={S.length - 1} value={i} onChange={(e) => setSnapId(S[+e.target.value].id)} />
        <span>快照 #{S[i].id} · 第 {S[i].step} 步 · 验证 {pct(S[i].val_acc)}</span>
        {snap?.separation != null && <span className="chip" title="类间距离 ÷ 类内距离，越大说明网络把各类分得越开">分离度 <b>{snap.separation}</b></span>}
        {snap?.dim && <span className="muted">向量 {snap.dim} 维 → PCA 2 维（解释 {P ? (100 * (P.explained[0] + P.explained[1])).toFixed(0) : 0}%）</span>}
      </div>
      <div className="space-main">
        <EmbedPlot P={P} />
        <div className="records">
          <div className="muted small-txt">向量记录（最难的几条）：id · vector · metadata</div>
          {snap?.records?.map((r, k) => (
            <pre key={r.id} className={"rec" + (k === rec ? " on" : "")} onClick={() => setRec(k)}>{JSON.stringify(k === rec ? r : { id: r.id, label: r.metadata.label, pred: r.metadata.pred, correct: r.metadata.correct }, null, k === rec ? 1 : 0)}</pre>
          ))}
        </div>
      </div>
    </div>
  );
}

function EmbedPlot({ P, size = 230 }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr; cv.height = size * dpr;
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, size, size);
    ctx.strokeStyle = "#262c3b"; ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
    if (!P?.points.length) return;
    const xs = P.points.map((p) => p[0]), ys = P.points.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), pad = 12;
    const sx = (v) => pad + ((v - x0) / (x1 - x0 || 1)) * (size - 2 * pad), sy = (v) => size - pad - ((v - y0) / (y1 - y0 || 1)) * (size - 2 * pad);
    for (const [a, b, lab, , ok, split] of P.points) {
      ctx.fillStyle = classColor(lab); ctx.globalAlpha = split === "val" ? 1 : 0.55;
      ctx.beginPath(); ctx.arc(sx(a), sy(b), split === "val" ? 4 : 2.6, 0, Math.PI * 2); ctx.fill();
      if (!ok) { ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.4; ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
  }, [P, size]);
  return (
    <div className="embed-plot">
      <canvas ref={ref} style={{ width: size, height: size }} />
      <div className="legend">{P?.classes.map((c, k) => <span key={c}><i style={{ background: classColor(k) }} />{c}</span>)}
        <span className="muted">大点 = 验证集 · 白圈 = 预测错</span></div>
    </div>
  );
}

// ---- 每层变化热力图：哪一层在什么时候学得最多 / per-layer change heat map ---------------------------
function Layers({ detail }) {
  if (!detail?.snapshots.length) return <Empty />;
  const { names, delta } = detail.layers;
  const all = delta.flat().filter((v) => v != null && v > 0);
  if (!all.length) return <Empty msg="至少要两个快照才能看出每层变化。" />;
  const lo = Math.log10(Math.min(...all)), hi = Math.log10(Math.max(...all));
  const color = (v) => { if (v == null || v <= 0) return "transparent"; const t = (Math.log10(v) - lo) / (hi - lo || 1); return `hsl(${250 - 210 * t} 75% ${28 + 32 * t}%)`; };
  const S = detail.snapshots;
  return (
    <div className="layers-map">
      <div className="muted small-txt">每格 = 该层权重在这段时间里变化了多少（‖W<sub>t</sub> − W<sub>t−1</sub>‖，对数色阶）：亮 = 正在大幅学习，暗 = 基本学完/冻结</div>
      <table>
        <tbody>
          {names.map((n, i) => (
            <tr key={n}><th>{n}</th>{delta[i].map((v, j) => <td key={j} style={{ background: color(v) }} title={`${n} · 第${S[j].step}步 · 变化 ${fmt(v)} · 梯度 ${fmt(detail.layers.gnorm[i][j])}`} />)}</tr>
          ))}
          <tr><th /><td colSpan={S.length} className="muted small-txt">第 {S[0].step} 步 → 第 {S[S.length - 1].step} 步</td></tr>
        </tbody>
      </table>
    </div>
  );
}

// ---- 样本记忆：长期累计，每个样本被看过 / 错过几次 / sample-level long-term memory ----------------
function Samples({ dataset, tick }) {
  const [items, setItems] = useState(null);
  useEffect(() => {
    if (!dataset) { setItems(null); return; }
    getJSON("/api/memory/samples?dataset=" + encodeURIComponent(dataset)).then((r) => setItems(r.items)).catch(() => setItems([]));
  }, [dataset, tick]);
  if (!dataset) return <Empty msg="样本记忆只在导入自己的数据时记录。" />;
  if (!items) return <Empty msg="读取中…" />;
  const forget = async () => {
    if (!confirm(tr("清空这份数据的样本记忆（错题本）？"))) return;
    await post("/api/memory/action", { action: "forget_samples", dataset }); setItems([]);
  };
  return (
    <div className="samples-mem">
      <div className="muted small-txt">跨多次训练累计的“错题本”：难例回放会按错误率和平滑损失加大这些样本的抽样权重。
        <button className="btn small" onClick={forget}>清空错题本</button></div>
      <div className="table-wrap">
        <table className="cols">
          <thead><tr><th>来源</th><th>划分</th><th>标签</th><th>评估次数</th><th>错误</th><th>平滑损失</th><th>最近预测</th><th>置信度</th></tr></thead>
          <tbody>{items.map((m) => (
            <tr key={m.sample_id} className={m.wrong / m.seen > 0.5 ? "hard" : ""}>
              <td title={m.source}>{m.source}</td><td>{m.split === "val" ? "验证" : "训练"}</td><td>{m.label}</td><td>{m.seen}</td>
              <td>{m.wrong} <span className="muted">({((100 * m.wrong) / Math.max(1, m.seen)).toFixed(0)}%)</span></td>
              <td>{fmt(m.ema_loss)}</td><td className={m.last_pred === m.label ? "ok" : "bad"}>{m.last_pred}</td><td>{m.last_conf == null ? "—" : pct(m.last_conf)}</td>
            </tr>))}</tbody>
        </table>
      </div>
    </div>
  );
}

// ---- 数据结构与只读 SQL 控制台 / schema + read-only SQL console ------------------------------------
function Schema({ ov }) {
  const [sql, setSql] = useState(PRESETS[0][1]);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState("");
  const runQ = async (q = sql) => {
    setErr("");
    try { setRes(await post("/api/memory/action", { action: "query", sql: q })); } catch (e) { setErr(e.message); setRes(null); }
  };
  return (
    <div className="schema">
      <div className="schema-tables">
        {ov?.tables.map((t) => <details key={t.name}><summary><code>{t.name}</code> <span className="muted">{ov.counts[t.name] ?? ""}</span></summary><pre>{t.sql}</pre></details>)}
        <div className="muted small-txt">文件：{ov?.path}</div>
      </div>
      <div className="sql">
        <div className="presets">{PRESETS.map(([t, q]) => <button key={t} className="chip" onClick={() => { setSql(q); runQ(q); }}>{t}</button>)}</div>
        <textarea value={sql} onChange={(e) => setSql(e.target.value)} rows={3} spellCheck={false}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runQ(); }} />
        <div><button className="btn small primary" onClick={() => runQ()}>运行（Ctrl+Enter）</button> <span className="muted small-txt">只读：只能 SELECT 查询，不会改动记忆</span></div>
        {err && <div className="err">⚠ {err}</div>}
        {res && (
          <div className="table-wrap">
            <table className="cols">
              <thead><tr>{res.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{res.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{v == null ? <span className="muted">NULL</span> : String(v)}</td>)}</tr>)}</tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function Empty({ msg = "选择一次训练运行，或开始训练后这里会实时出现记忆。" }) { return <div className="empty">{msg}</div>; }
