// 数据导入与分析模块 / Data import & analysis
//  ① 选择数据：本地路径（文件或文件夹）、上传文件 / 上传整个文件夹 / 拖进来（保留类别子文件夹）
//     图片和音频的标签可以来自：元数据表（labels.csv: filename,label）、类别子文件夹、文件名前缀
//  ② 自动识别并分析：列类型、缺失、分布、目标分布、特征相关性、PCA 二维投影、样本预览
//  ③ 一键用这份数据训练（自动选择 MLP 或 ViT），训练后可做完整的模型表现分析
import { useEffect, useRef, useState } from "react";
import { apiBase, apiFetch, fmt, getJSON, readJSON } from "../api.js";
import { Bars, classColor, Confusion, Hist, Lines, Scatter, Thumb } from "./viz.jsx";

async function post(url, body) {
  const r = await apiFetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return readJSON(r);
}

const TYPE_CN = { numeric: "数值", categorical: "类别", text: "文本", id: "ID" };
const KIND_CN = { tabular: "表格", image: "图片", audio: "音频", array: "数组" };
const SRC_CN = { metadata: "元数据表", folders: "类别子文件夹", filename: "文件名前缀" };
const EXT = (n) => (n.match(/\.[^./\\]+$/)?.[0] || "").toLowerCase();
const SINGLE = new Set([".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet", ".npz", ".npy", ".zip"]);
const MEDIA = new Set([".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff",
  ".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aiff", ".aif"]);
const ACCEPT = [...SINGLE, ...MEDIA].join(",");

// 拖入文件夹时递归读出所有文件及其相对路径 / walk dropped folders
async function filesFromDrop(dt) {
  const entries = [...dt.items].map((it) => it.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dt.files].map((f) => ({ file: f, rel: f.name }));
  const out = [];
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const f = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ file: f, rel: prefix + f.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const e of batch) await walk(e, prefix + entry.name + "/");
      } while (batch.length);
    }
  };
  for (const e of entries) await walk(e, "");
  return out;
}

export default function DataView({ onTrain, customRunning, log }) {
  const [path, setPath] = useState("");
  const [info, setInfo] = useState(null);
  const [opts, setOpts] = useState({ target: "", task_type: "auto", val_ratio: 0.2, img_size: 32, max_per_class: 1000 });
  const [analysis, setAnalysis] = useState(null);
  const [busy, setBusy] = useState("");
  const [err, setErr] = useState("");
  const [evalRes, setEvalRes] = useState(null);
  const [backend, setBackend] = useState(null);       // null=检查中, {version}=正常, {stale:true}=旧后端
  const [drag, setDrag] = useState(false);
  const [uploads, setUploads] = useState([]);           // data/uploads 下已上传的数据
  const [batch, setBatchState] = useState(() => { try { return localStorage.getItem("nc.batch") || ""; } catch { return ""; } });
  const setBatch = (b) => { setBatchState(b); try { localStorage.setItem("nc.batch", b); } catch { /* 忽略 */ } };
  const dirInput = useRef(null);
  const refreshUploads = () => getJSON("/api/data/uploads").then((r) => setUploads(r.items || [])).catch(() => {});

  useEffect(() => {
    if (location.protocol === "file:") { setBackend({ stale: true, file: true }); return; }
    getJSON("/api/version").then((v) => setBackend(v.version >= 3 ? v : { ...v, stale: true })).catch(() => setBackend({ stale: true }));
    refreshUploads();
    getJSON("/api/data/current").then((a) => { if (a?.n) setAnalysis(a); }).catch(() => {});
  }, []);
  useEffect(() => { dirInput.current?.setAttribute("webkitdirectory", ""); }, []);

  const run = async (label, fn) => {
    setBusy(label); setErr("");
    try { return await fn(); } catch (e) { setErr(e.message || String(e)); } finally { setBusy(""); }
  };
  const applyInfo = (d) => {
    setInfo(d); setPath(d.path);
    setOpts((o) => ({ ...o, target: d.suggested_target || "" }));
  };
  // 识别成功且可以训练时，自动“加载并分析”，不用再找按钮
  const analyzeNow = async (p, o) => {
    setBusy("读取并分析中…（图片多时需要几秒）");
    const a = await post("/api/data/load", { path: p, ...o });
    setAnalysis(a); setEvalRes(null);
    log?.(`数据已载入：${a.n} 条，${a.task_type === "regression" ? "回归" : a.classes.length + " 类分类"}`);
    setTimeout(() => document.getElementById("analysis-card")?.scrollIntoView({ behavior: "smooth" }), 50);
  };
  const inspectAndAnalyze = async (p) => {
    setBusy("识别中…");
    const d = await post("/api/data/inspect", { path: p });
    applyInfo(d); setAnalysis(null);
    if (d.ready !== false && d.kind !== "tabular") await analyzeNow(d.path, { ...opts, target: d.suggested_target || "" });
  };
  const inspect = (p = path, target) => run("识别中…", async () => {
    const d = await post("/api/data/inspect", { path: p, target });
    if (target) { setInfo(d); setOpts((o) => ({ ...o, target })); } else applyInfo(d);
    if (p.includes("uploads")) setBatch(d.path.split(/[\\/]/).slice(-1)[0]);
  });
  const pickUpload = (u) => run("识别中…", async () => {
    setPath(u.path);
    if (u.is_dir) setBatch(u.name);
    await inspectAndAnalyze(u.path);
  });

  // 上传：一个表格/数组/zip 直接传；多个文件或整个文件夹按“批次”逐个上传并保留子文件夹
  const uploadList = (list) => run("准备上传…", async () => {
    list = list.filter(({ file, rel }) => !/(^|\/)(\.|__MACOSX)/.test(rel) && file.size > 0);
    if (!list.length) throw new Error("没有可上传的文件。");
    const known = list.filter(({ file }) => SINGLE.has(EXT(file.name)) || MEDIA.has(EXT(file.name)));
    if (!known.length) throw new Error("不支持这些文件类型：" + [...new Set(list.map(({ file }) => EXT(file.name) || file.name))].slice(0, 6).join(" "));
    let target;
    if (known.length === 1 && SINGLE.has(EXT(known[0].file.name))) {
      const f = known[0].file;
      setBusy(`上传 ${f.name}…`);
      const j = await readJSON(await apiFetch("/api/data/upload?name=" + encodeURIComponent(f.name), { method: "POST", body: f }));
      log?.(`已上传 ${f.name}（${(j.size / 1024).toFixed(0)} KB）`);
      target = j.path;
    } else {
      // 继续加入“当前数据集”：先传 cat 文件夹、再传 dog 文件夹，会放在同一个批次里 → 2 个类别
      const cur = batch || new Date().toISOString().replace(/\D/g, "").slice(0, 14) + "_" + Math.random().toString(36).slice(2, 6);
      if (!batch) setBatch(cur);
      let done = 0, bytes = 0, root = null;
      const queue = [...known];
      const worker = async () => {
        while (queue.length) {
          const { file, rel } = queue.shift();
          const j = await readJSON(await apiFetch(`/api/data/upload?batch=${cur}&rel=${encodeURIComponent(rel)}`, { method: "POST", body: file }));
          root = j.path; bytes += j.size; done += 1;
          setBusy(`上传中 ${done} / ${known.length} 个文件（${(bytes / 1048576).toFixed(1)} MB）…`);
        }
      };
      await Promise.all([1, 2, 3, 4].map(worker));
      log?.(`已上传 ${known.length} 个文件（${(bytes / 1048576).toFixed(1)} MB）`);
      if (known.length < list.length) log?.(`跳过 ${list.length - known.length} 个不支持的文件`);
      target = root;
    }
    refreshUploads();
    await inspectAndAnalyze(target);
  });
  const fromInput = (e) => {
    const fs = [...(e.target.files || [])].map((f) => ({ file: f, rel: f.webkitRelativePath || f.name }));
    e.target.value = "";
    if (fs.length) uploadList(fs);
  };
  const onDrop = async (e) => {
    e.preventDefault(); setDrag(false);
    try { uploadList(await filesFromDrop(e.dataTransfer)); } catch (x) { setErr(String(x)); }
  };

  const load = () => run("读取并分析中…", () => analyzeNow(path, opts));
  const cur = uploads.find((u) => u.name === batch);
  const evaluate = () => run("分析模型中…", async () => setEvalRes(await post("/api/data/evaluate", {})));
  const media = info && (info.kind === "image" || info.kind === "audio");
  const nClasses = info?.classes ? Object.keys(info.classes).length : 0;

  return (
    <div className="dataview">
      {backend?.stale && (
        <div className="err banner">⚠ {backend.file
          ? "这个页面是直接双击 HTML 打开的，无法连接后端。请运行 .\\scripts\\run.ps1 -Task ui，然后在浏览器打开 http://127.0.0.1:8765"
          : apiBase() ? `连不上后端 ${apiBase()}，或它是旧版本。点右上角「后端」检查地址；用本机后端时先运行 .\\scripts\\run.ps1 -Task ui。`
          : "当前运行的后端是旧版本（没有新的数据接口），上传会失败。请关闭正在运行后端的 PowerShell 窗口（或按 Ctrl+C），再运行 .\\scripts\\run.ps1 -Task ui，然后刷新本页。"}</div>
      )}
      <section className={"card" + (drag ? " dragging" : "")}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}>
        <div className="card-head"><span>① 选择数据</span><span className="muted small-txt">所有数据只在你本机处理 · 也可以把文件 / 文件夹直接拖进这个框</span></div>
        <div className="src-row">
          {backend?.cloud ? <span className="path cloud-note">☁ 云端后端：请上传文件或文件夹（不能读取服务器本地路径）</span> : <>
          <input className="path" placeholder="本地路径：D:\数据\iris.csv 或 D:\数据\图片文件夹" value={path}
            onChange={(e) => setPath(e.target.value)} onKeyDown={(e) => e.key === "Enter" && inspect()} />
          <button className="btn" disabled={!path || !!busy} onClick={() => inspect()}>识别</button></>}
          <label className={"btn upload" + (busy ? " disabled" : "")}>上传文件
            <input type="file" multiple accept={ACCEPT} disabled={!!busy} onChange={fromInput} />
          </label>
          <label className={"btn upload" + (busy ? " disabled" : "")} title="选择整个文件夹（例如按类别分好子文件夹的图片/音频，或含 labels.csv 的文件夹）">上传文件夹
            <input type="file" ref={dirInput} multiple disabled={!!busy} onChange={fromInput} />
          </label>
        </div>
        <div className="formats">
          <span><b>表格</b> CSV / TSV / JSON / Excel(.xlsx) / Parquet —— 每行一个样本，选一列作为要预测的目标</span>
          <span><b>数组</b> .npz / .npy（X 和 y；或 x_train / y_train / x_test / y_test；或一个二维数组，最后一列是目标）</span>
          <span><b>图片</b> 上传整个文件夹或 zip；标签来自：类别子文件夹（猫\1.jpg）、文件名前缀（cat_1.jpg）或元数据表（labels.csv：filename,label）</span>
          <span><b>音频</b> .wav（.flac/.ogg/.mp3 需 pip install soundfile），标签方式同图片；元数据里的数值标签可做回归</span>
        </div>
        <div className="uploaded">
          <div className="uploaded-head">
            <b>已上传的数据</b>
            <span className="muted small-txt">多次上传的文件夹会合并进「当前数据集」：先传 cat 文件夹再传 dog 文件夹 = 2 个类别</span>
            <button className="btn small" disabled={!!busy} onClick={() => { setBatch(""); setInfo(null); setAnalysis(null); setPath(""); }}
              title="下一次上传会放进一个新的数据集">＋ 新建数据集</button>
            {backend?.version && <span className="ver">后端 v{backend.version}</span>}
          </div>
          {cur ? (
            <div className="cur-ds" onClick={() => pickUpload(cur)} title="点击重新识别并分析">
              <span className="badge big">当前数据集</span>
              <code>{cur.name}</code>
              <span className="tree">{Object.entries(cur.tree).map(([k, v]) => <span key={k} className="chip">📁 {k} <b>{v}</b></span>)}</span>
              <span className="muted">共 {cur.files} 个文件</span>
            </div>
          ) : <div className="muted small-txt">{batch ? "当前数据集：" + batch + "（还没有文件）" : "还没有选择数据集——上传文件或文件夹后会出现在这里。"}</div>}
          {uploads.length > 0 && (
            <details className="history">
              <summary>以前上传的（{uploads.length}）——点一个即可重新使用</summary>
              {uploads.map((u) => (
                <div key={u.name} className={"hist-row" + (u.name === batch ? " on" : "")} onClick={() => pickUpload(u)}>
                  <code>{u.name}</code>
                  <span className="tree">{Object.entries(u.tree).slice(0, 8).map(([k, v]) => <span key={k} className="chip">{k} {v}</span>)}</span>
                  <span className="muted">{u.files} 个文件 · {new Date(u.mtime * 1000).toLocaleString()}</span>
                </div>
              ))}
            </details>
          )}
        </div>
        {info && info.ready === false && (
          <div className="hint">ℹ {info.problem}
            {info.tree && <div className="tree">{Object.entries(info.tree).map(([k, v]) => <span key={k} className="chip">📁 {k} <b>{v}</b></span>)}</div>}
          </div>
        )}
        {info && info.ready !== false && (
          <div className="inspect">
            <span className="badge big">{KIND_CN[info.kind]}</span>
            {info.kind === "tabular" && (
              <>
                <label>目标列（要预测的）
                  <select value={opts.target} onChange={(e) => setOpts({ ...opts, target: e.target.value })}>
                    {info.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <span className="muted">{info.columns.length} 列</span>
              </>
            )}
            {media && (
              <>
                <span className="muted">{info.files} 个文件 · 标签来源：<b className="hl">{SRC_CN[info.label_source]}</b>
                  {info.meta_file ? `（${info.meta_file}，按「${info.file_column}」列匹配文件名）` : ""}</span>
                {info.columns && (
                  <label>标签列
                    <select value={opts.target} onChange={(e) => inspect(info.path, e.target.value)}>
                      {info.columns.filter((c) => c !== info.file_column).map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </label>
                )}
                <span className="muted">{info.numeric_target ? "数值标签（回归）"
                  : `${nClasses} 个类别：` + Object.entries(info.classes).slice(0, 12).map(([k, v]) => `${k}(${v})`).join("，") + (nClasses > 12 ? " …" : "")}
                  {info.has_split ? "（含 train/val 划分）" : ""}{info.unmatched ? `；${info.unmatched} 个文件在表里没有对应行` : ""}</span>
              </>
            )}
            {info.kind === "array" && <span className="muted">{Object.entries(info.arrays).map(([k, v]) => `${k}[${v.join("×")}]`).join("  ")}</span>}
            <label>任务
              <select value={opts.task_type} onChange={(e) => setOpts({ ...opts, task_type: e.target.value })}>
                <option value="auto">自动判断</option><option value="classification">分类</option><option value="regression">回归（预测数值）</option>
              </select>
            </label>
            <label>验证集比例<input type="number" step="0.05" min="0.05" max="0.5" value={opts.val_ratio}
              onChange={(e) => setOpts({ ...opts, val_ratio: +e.target.value })} /></label>
            {info.kind === "image" && (
              <label>图片尺寸<select value={opts.img_size} onChange={(e) => setOpts({ ...opts, img_size: +e.target.value })}>
                {[16, 32, 48, 64].map((s) => <option key={s} value={s}>{s}×{s}</option>)}</select></label>
            )}
            {media && (
              <label>每类最多<input type="number" min="10" value={opts.max_per_class} onChange={(e) => setOpts({ ...opts, max_per_class: +e.target.value })} /></label>
            )}
            <button className="btn primary" disabled={!!busy} onClick={load}>{analysis ? "按新选项重新分析" : "加载并分析 ▶"}</button>
          </div>
        )}
        {busy && <div className="busy">{busy}</div>}
        {err && <div className="err">⚠ {err}</div>}
      </section>

      {analysis && <Analysis a={analysis} onTrain={() => onTrain(analysis)} />}

      {analysis && (
        <section className="card">
          <div className="card-head"><span>③ 模型表现分析</span>
            <button className="btn primary" disabled={!customRunning || !!busy} onClick={evaluate}
              title={customRunning ? "" : "先用这份数据训练一会儿"}>分析模型表现</button></div>
          {!evalRes ? <div className="empty">{customRunning ? "训练一段时间后点击右上角按钮：给出准确率/误差、混淆矩阵、每类指标、特征重要性和错得最离谱的样本。" : "先点上面的「用这份数据训练」，在训练页开始训练后回到这里。"}</div>
            : <EvalReport r={evalRes} />}
        </section>
      )}
    </div>
  );
}

function Stat({ v, k }) { return <div className="stat"><b>{v}</b><span>{k}</span></div>; }

function Analysis({ a, onTrain }) {
  const isCls = a.task_type === "classification";
  const [colFilter, setColFilter] = useState("");
  return (
    <section className="card" id="analysis-card">
      <div className="card-head"><span>② 数据分析 <small className="muted">{a.source}</small></span>
        <button className="btn primary" onClick={onTrain}>▶ 用这份数据训练</button></div>
      <div className="go-train">
        ✅ 数据已准备好：{a.n} 个样本{a.task_type === "classification" ? `，${a.classes.length} 个类别（${a.classes.slice(0, 6).join("、")}${a.classes.length > 6 ? "…" : ""}）` : "，回归"}，模型 {a.model}。
        <button className="btn primary big" onClick={onTrain}>▶ 用这份数据训练（跳到训练可视化）</button>
      </div>
      <div className="stats six">
        <Stat v={a.n.toLocaleString()} k="样本数" />
        <Stat v={`${a.n_train} / ${a.n_val}`} k="训练 / 验证" />
        <Stat v={a.input_shape.length > 1 ? a.input_shape.join("×") : a.n_features} k={a.input_shape.length > 1 ? "输入形状" : "特征数（编码后）"} />
        <Stat v={isCls ? `${a.classes.length} 类` : "回归"} k={`目标：${a.target}`} />
        <Stat v={KIND_CN[a.kind]} k="数据类型" />
        <Stat v={<span className="small-model">{a.model}</span>} k="自动选择的模型" />
      </div>
      {(a.notes?.length > 0 || a.dropped?.length > 0) && (
        <ul className="notes">
          {a.notes.map((n, i) => <li key={i}>{n}</li>)}
          {a.dropped.map((d, i) => <li key={"d" + i}>已忽略列 <b>{d.name}</b>：{d.reason}</li>)}
        </ul>
      )}
      <div className="grid2">
        <div className="sub">
          <h4>目标分布 · {a.target}</h4>
          {a.target_dist.type === "classes"
            ? <Bars items={a.classes.map((c, i) => ({ label: c, value: a.target_dist.counts[i], color: classColor(i) }))} format={(v) => v} labelWidth={110} />
            : <><Hist hist={a.target_dist} width={320} height={90} /><div className="muted">范围 {fmt(a.target_dist.min)} ~ {fmt(a.target_dist.max)}，均值 {fmt(a.target_dist.mean)} ± {fmt(a.target_dist.std)}</div></>}
        </div>
        <div className="sub">
          <h4>PCA 二维投影 <small className="muted">解释方差 {(a.pca.explained[0] * 100).toFixed(0)}% + {(a.pca.explained[1] * 100).toFixed(0)}%</small></h4>
          <Scatter points={a.pca.points} mode={isCls ? "class" : "value"} size={260} xLabel="主成分 1" yLabel="主成分 2" />
          <div className="muted small-txt">把所有特征压缩到 2 维看样本分布：{isCls ? "不同颜色 = 不同类别，越分得开越容易学。" : "颜色 = 目标值大小。"}</div>
        </div>
      </div>
      {a.relevance && (
        <div className="sub">
          <h4>特征与目标的相关程度 <small className="muted">{isCls ? "相关比 η²（类间差异占总方差的比例）" : "|皮尔逊相关系数|"}，前 20</small></h4>
          <Bars items={a.relevance.map((r) => ({ label: r.name, value: r.score }))} max={1} labelWidth={190} />
        </div>
      )}
      {a.columns?.length > 0 && (
        <div className="sub">
          <h4>每一列的情况 <input className="filter" placeholder="筛选列名" value={colFilter} onChange={(e) => setColFilter(e.target.value)} /></h4>
          <div className="table-wrap">
            <table className="cols">
              <thead><tr><th>列名</th><th>类型</th><th>缺失</th><th>不同取值</th><th>统计</th><th>分布</th></tr></thead>
              <tbody>
                {a.columns.filter((c) => !colFilter || c.name.includes(colFilter)).map((c) => (
                  <tr key={c.name} className={c.name === a.target ? "target" : ""}>
                    <td>{c.name}{c.name === a.target && <span className="badge">目标</span>}</td>
                    <td><span className={"tbadge " + c.type}>{TYPE_CN[c.type] || c.type}</span></td>
                    <td>{c.missing || ""}</td><td>{c.unique}</td>
                    <td className="muted">{c.type === "numeric" ? `${fmt(c.mean, 3)} ± ${fmt(c.std, 3)}  [${fmt(c.min, 2)}, ${fmt(c.max, 2)}]`
                      : (c.top || []).slice(0, 3).map(([k, n]) => `${k}(${n})`).join("  ")}</td>
                    <td>{c.type === "numeric" ? <Hist hist={c.hist} /> : <Hist hist={{ counts: (c.top || []).map((t) => t[1]) }} color="#bb9af7" />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {a.samples && (
        <div className="sub">
          <h4>样本预览</h4>
          {a.samples.type === "rows" && (
            <div className="table-wrap"><table className="cols"><thead><tr>{a.samples.header.map((h) => <th key={h}>{h}</th>)}</tr></thead>
              <tbody>{a.samples.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div>
          )}
          {a.samples.type === "images" && (
            <div className="grid-imgs">{a.samples.items.map((it, i) => (
              <figure key={i}><Thumb {...it} scale={64 / it.w} /><figcaption>{it.label}</figcaption></figure>))}</div>
          )}
          {a.samples.type === "spectra" && (
            <>
              <div className="muted small-txt">每个类别的平均频谱（{a.samples.band_hz.length} 个梅尔频带的 log 能量）</div>
              <Lines series={a.samples.classes.map((c, i) => ({ values: c.mean, color: classColor(i) }))} labels={a.samples.band_hz.map((h) => h + "Hz")} />
              <div className="legend">{a.samples.classes.map((c, i) => <span key={c.name}><i style={{ background: classColor(i) }} />{c.name}</span>)}</div>
            </>
          )}
        </div>
      )}
    </section>
  );
}

function EvalReport({ r }) {
  const isCls = r.task_type === "classification";
  return (
    <div className="eval">
      <div className="stats six">
        {isCls ? <Stat v={(r.acc * 100).toFixed(1) + "%"} k="验证集准确率" /> : <>
          <Stat v={fmt(r.r2, 3)} k="R²（1 = 完美）" /><Stat v={fmt(r.mae, 4)} k="平均绝对误差 MAE" /><Stat v={fmt(r.rmse, 4)} k="均方根误差 RMSE" /></>}
        <Stat v={r.n_val} k="验证样本" />
      </div>
      <div className="grid2">
        {isCls ? (
          <div className="sub"><h4>混淆矩阵</h4><Confusion matrix={r.confusion} classes={r.classes} /></div>
        ) : (
          <div className="sub"><h4>残差分布（预测 − 真实）</h4><Hist hist={r.residual_hist} width={320} height={90} color="#e0af68" />
            <div className="muted">范围 {fmt(r.residual_hist.min)} ~ {fmt(r.residual_hist.max)}</div></div>
        )}
        {r.importance && (
          <div className="sub"><h4>特征重要性 <small className="muted">置换法：打乱该特征后{r.importance_metric}</small></h4>
            <Bars items={r.importance.map((d) => ({ label: d.name, value: d.drop }))} labelWidth={170} color="#e0af68" />
            {r.importance_note && <div className="muted small-txt">{r.importance_note}</div>}</div>
        )}
      </div>
      {isCls && r.per_class && (
        <div className="sub"><h4>每个类别的指标</h4>
          <table className="cols"><thead><tr><th>类别</th><th>精确率</th><th>召回率</th><th>F1</th><th>样本数</th></tr></thead>
            <tbody>{r.per_class.map((c, i) => <tr key={i}><td><i className="dot" style={{ background: classColor(i) }} />{c.name}</td>
              <td>{fmt(c.precision, 3)}</td><td>{fmt(c.recall, 3)}</td><td>{fmt(c.f1, 3)}</td><td>{c.support}</td></tr>)}</tbody></table></div>
      )}
      {r.worst?.length > 0 && (
        <div className="sub"><h4>{isCls ? "最自信的错误（模型很确定但答错了）" : "误差最大的样本"}</h4>
          {r.worst[0].px ? (
            <div className="grid-imgs">{r.worst.map((it, i) => <figure key={i}><Thumb {...it} scale={64 / it.w} />
              <figcaption className="bad">预测 {it.pred} {Math.round(it.conf * 100)}%<br /><small className="muted">真实 {it.label}</small></figcaption></figure>)}</div>
          ) : (
            <table className="cols"><thead><tr><th>样本行号</th>{isCls ? <><th>真实</th><th>预测</th><th>置信度</th></> : <><th>真实值</th><th>预测值</th></>}</tr></thead>
              <tbody>{r.worst.map((w, i) => <tr key={i}><td>#{w.index}</td>{isCls ? <><td>{w.label}</td><td className="bad">{w.pred}</td><td>{fmt(w.conf, 3)}</td></>
                : <><td>{fmt(w.target)}</td><td className="bad">{fmt(w.pred)}</td></>}</tr>)}</tbody></table>
          )}
        </div>
      )}
    </div>
  );
}
