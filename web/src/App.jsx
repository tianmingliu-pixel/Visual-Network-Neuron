import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { control, fmt, getJSON, useStream } from "./api.js";
import LineChart from "./components/LineChart.jsx";
import Preview from "./components/Preview.jsx";
import CodePanel from "./components/CodePanel.jsx";
import TracePlayer from "./components/TracePlayer.jsx";
import Explain from "./components/Explain.jsx";
import NetworkGraph from "./components/NetworkGraph.jsx";
import Pipeline from "./components/Pipeline.jsx";
import DataView from "./components/DataView.jsx";
import MemoryPanel from "./components/MemoryPanel.jsx";
import BackendSettings from "./components/BackendSettings.jsx";

// 图表系列（放在组件外，避免每次渲染都重画）/ chart series defined once
const LOSS_SERIES = [
  { key: "loss", label: "loss", color: "#7aa2f7", alpha: 0.35, width: 1 },
  { key: "loss", label: "平滑", color: "#7aa2f7", smooth: 0.05, width: 2 },
];
const AUX_SERIES = [
  { key: "acc", label: "准确率", color: "#9ece6a" },
  { key: "grad_norm", label: "梯度范数", color: "#e0af68" },
];
const MAX_POINTS = 50000;
const STATE_CN = { idle: "空闲", running: "训练中", paused: "已暂停", finished: "已完成", stopped: "已停止", error: "出错" };

export default function App() {
  const [tasks, setTasks] = useState([]);
  const [taskId, setTaskId] = useState("mlp");
  const [params, setParams] = useState({ steps: 400, lr: 0.001, batch_size: 64 });
  const [status, setStatus] = useState({ state: "idle", step: 0, total: 0 });
  const [preview, setPreview] = useState(null);
  const [graph, setGraph] = useState(null);
  const [pin, setPin] = useState(null);          // 从结构图/流程图点击跳转的代码行 / pinned code line
  const [logs, setLogs] = useState([]);
  const [busy, setBusy] = useState(false);
  // 机器记忆库 / memory store options for the next run
  const [memOpts, setMemOpts] = useState({ memory: true, resume: null, replay: true, augment: true, mem_every: 0 });
  const [memTick, setMemTick] = useState(0);
  const [bottom, setBottom] = useState("memory");   // 右下角：memory = 记忆库，explain = 逐行讲解

  // 指标放在 ref 里，按动画帧合并刷新，避免每个数据点都触发 React 渲染
  const metricsRef = useRef([]);
  const [mv, setMv] = useState(0);
  const raf = useRef(0);
  const bump = () => {
    if (!raf.current) raf.current = requestAnimationFrame(() => { raf.current = 0; setMv((v) => v + 1); });
  };

  // 逐行轨迹 / trace playback state
  const [trace, setTrace] = useState(null);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [quiet, setQuiet] = useState(false);
  const [sources, setSources] = useState({});
  const [activeFile, setActiveFile] = useState(null);
  const [follow, setFollow] = useState(true);
  const [showComments, setShowComments] = useState(true);

  const log = useCallback((msg, level = "info") => {
    setLogs((l) => [...l.slice(-60), { t: new Date().toLocaleTimeString(), msg, level }]);
  }, []);

  const conn = useStream({
    status: setStatus,
    metrics: (d) => {
      const arr = metricsRef.current;
      for (const p of d.points) arr.push(p);
      if (arr.length > MAX_POINTS) arr.splice(0, arr.length - MAX_POINTS);
      bump();
    },
    preview: setPreview,
    graph: setGraph,
    log: (d) => log(d.msg, d.level),
    reset: () => { metricsRef.current = []; setPreview(null); setGraph(null); setTrace(null); bump(); },
    trace: (d) => { setTrace(d); setPos(0); setPlaying(true); setFollow(true); setBottom("explain"); },
    memory: (d) => { setMemTick((t) => t + 1); if (d.event === "snapshot" && d.snapshot?.is_best) log(`🧠 新的最佳记忆：第 ${d.snapshot.step} 步，验证 ${(d.snapshot.val_acc * 100).toFixed(1)}%`); },
  });

  const [view, setView] = useState("train");     // train = 训练可视化，data = 数据导入与分析
  const loadTasks = useCallback((select) => getJSON("/api/tasks").then((ts) => {
    setTasks(ts);
    const t = ts.find((x) => x.id === select) || ts.find((x) => x.id === "mlp") || ts[0];
    if (t) { setTaskId(t.id); setParams(t.defaults); }
  }).catch((e) => log("无法读取任务列表: " + e.message, "error")), [log]);
  useEffect(() => { loadTasks(); }, [loadTasks]);
  const trainOnData = async () => { await loadTasks("custom"); setView("train"); };

  const task = tasks.find((t) => t.id === taskId);
  const pickTask = (id) => {
    setTaskId(id);
    const t = tasks.find((x) => x.id === id);
    if (t) setParams(t.defaults);
  };

  // 加载源代码（带缓存）/ load annotated sources
  const requested = useRef(new Set());
  const ensureSource = useCallback((f) => {
    if (requested.current.has(f)) return;
    requested.current.add(f);
    getJSON("/api/source?file=" + encodeURIComponent(f))
      .then((src) => setSources((x) => ({ ...x, [f]: src })))
      .catch((e) => { requested.current.delete(f); log(`读取 ${f} 失败: ${e.message}`, "error"); });
  }, [log]);

  const files = useMemo(() => {
    const set = new Set(task?.files || []);
    for (const f of trace?.files || []) set.add(f);
    return [...set];
  }, [task, trace]);

  useEffect(() => { for (const f of files) ensureSource(f); }, [files, ensureSource]);
  useEffect(() => { if (!activeFile || !files.includes(activeFile)) setActiveFile(files[1] || files[0] || null); }, [files, activeFile]);

  // 预处理轨迹：调用栈路径 + 每个文件执行过的行 / precompute call paths & visited lines
  const { order, paths, visited } = useMemo(() => {
    if (!trace) return { order: [], paths: [], visited: {} };
    const stack = [], paths = [], visited = {};
    trace.events.forEach((ev, i) => {
      const d = ev.d || 1;
      stack.length = Math.min(stack.length, d - (ev.t === "call" ? 1 : 0));
      if (ev.t === "call") stack.push((ev.c ? ev.c + "." : "") + ev.fn);
      paths[i] = [...stack];
      if (ev.f != null) {
        const f = trace.files[ev.f];
        (visited[f] ||= new Set()).add(ev.l);
      }
    });
    const order = trace.events.map((_, i) => i).filter((i) => {
      const ev = trace.events[i];
      return !quiet || ev.t !== "line" || (ev.v && ev.v.length) || ev.ret;
    });
    return { order, paths, visited };
  }, [trace, quiet]);

  useEffect(() => { setPos((p) => Math.min(p, Math.max(0, order.length - 1))); }, [order]);

  // 自动播放 / autoplay
  useEffect(() => {
    if (!playing || !order.length) return;
    const id = setInterval(() => {
      setPos((p) => {
        if (p >= order.length - 1) { setPlaying(false); return p; }
        return p + 1;
      });
    }, 600 / speed);
    return () => clearInterval(id);
  }, [playing, speed, order.length]);

  const ev = trace && order.length ? trace.events[order[pos]] : null;

  // 每个轨迹事件属于训练的哪个阶段（数据/前向/损失/反向/更新）/ stage of each trace event
  const pipeline = task?.pipeline || [];
  const stageOf = useMemo(() => {
    if (!trace || !pipeline.length) return [];
    const byLine = new Map(), dataFns = new Set();
    for (const st of pipeline) {
      for (const r of st.refs) if (st.id !== "forward" || r.label === "调用网络") byLine.set(r.file + ":" + r.line, st.id);
      for (const f of st.fns || []) dataFns.add(f);
    }
    let prev = "data";
    return trace.events.map((e) => {
      const f = e.f != null ? trace.files[e.f] : "";
      let st = null;
      if (e.t === "note") st = "backward";
      else if (byLine.has(f + ":" + e.l)) st = byLine.get(f + ":" + e.l);
      else if (dataFns.has(e.fn)) st = "data";
      else if (f.startsWith("neurocore/models/") || f.startsWith("neurocore/layers/")) st = "forward";
      prev = st || prev;
      return prev;
    });
  }, [trace, pipeline]);

  // 没有回放时，训练中按节奏循环高亮各阶段（示意）/ decorative cycle while training
  const [animStage, setAnimStage] = useState(0);
  useEffect(() => {
    if (status.state !== "running" || (trace && order.length)) return;
    const id = setInterval(() => setAnimStage((k) => (k + 1) % 5), 650);
    return () => clearInterval(id);
  }, [status.state, trace, order.length]);
  let activeStage = null, stageSource = null;
  if (ev) { activeStage = stageOf[order[pos]]; stageSource = "trace"; }
  else if (status.state === "running" && pipeline.length) { activeStage = pipeline[animStage]?.id; stageSource = "anim"; }
  const pulse = !activeStage ? "idle" : (activeStage === "backward" || activeStage === "update") ? "backward" : "forward";

  const jump = (ref) => {
    if (!ref?.file) return;
    setFollow(false); setPlaying(false);
    ensureSource(ref.file);
    setActiveFile(ref.file);
    setPin({ ...ref, t: Date.now() });
  };
  const evFile = ev && ev.f != null ? trace.files[ev.f] : null;
  useEffect(() => { if (follow && evFile) setActiveFile(evFile); }, [evFile, follow]);
  useEffect(() => { if (follow) setPin(null); }, [follow]);

  // 键盘：空格播放/暂停，左右方向键单步 / keyboard shortcuts
  useEffect(() => {
    const onKey = (e) => {
      if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) || !order.length) return;
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
      if (e.key === "ArrowRight") { setPlaying(false); setPos((p) => Math.min(order.length - 1, p + 1)); }
      if (e.key === "ArrowLeft") { setPlaying(false); setPos((p) => Math.max(0, p - 1)); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [order.length]);

  const send = async (cmd) => {
    setBusy(true);
    try {
      const extra = cmd === "start" ? { ...memOpts, resume: memOpts.resume || undefined } : {};
      const r = await control({ cmd, task: taskId, ...params, ...extra });
      if (cmd === "start" && memOpts.resume && memOpts.resume !== "best") setMemOpts((o) => ({ ...o, resume: null }));
      if (r.path) log(`ONNX 已导出：${r.path}`);
      setStatus((s) => ({ ...s, ...r }));
    } catch (e) {
      log(`${cmd} 失败: ${e.message}`, "error");
    } finally {
      setBusy(false);
    }
  };

  const st = status.state;
  const running = st === "running";
  const sameTask = status.task === taskId;
  const last = metricsRef.current[metricsRef.current.length - 1];
  const pct = status.total ? Math.min(100, (status.step / status.total) * 100) : 0;

  return (
    <div className="app">
      <header className="bar">
        <div className="brand">NeuroCore</div>
        <div className="views">
          <button className={"view" + (view === "train" ? " on" : "")} onClick={() => setView("train")}>训练可视化</button>
          <button className={"view" + (view === "data" ? " on" : "")} onClick={() => setView("data")}>数据导入与分析</button>
        </div>
        <select value={taskId} onChange={(e) => pickTask(e.target.value)} disabled={running}>
          {tasks.map((t) => <option key={t.id} value={t.id}>{t.title}</option>)}
        </select>
        <label>步数<input type="number" min="1" value={params.steps} onChange={(e) => setParams({ ...params, steps: +e.target.value })} /></label>
        <label>学习率<input type="number" step="0.0001" value={params.lr} onChange={(e) => setParams({ ...params, lr: +e.target.value })} /></label>
        <label>批大小<input type="number" min="1" value={params.batch_size} onChange={(e) => setParams({ ...params, batch_size: +e.target.value })} /></label>
        <div className="actions">
          {!running && (st !== "paused" || !sameTask) && <button className="btn primary" disabled={busy} onClick={() => send("start")}>▶ 开始训练</button>}
          {running && <button className="btn" disabled={busy} onClick={() => send("pause")}>⏸ 暂停</button>}
          {st === "paused" && sameTask && <button className="btn primary" disabled={busy} onClick={() => send("resume")}>▶ 继续</button>}
          {(running || st === "paused") && <button className="btn" disabled={busy} onClick={() => send("stop")}>■ 停止</button>}
          {(st === "finished" || st === "stopped") && sameTask && <button className="btn" disabled={busy} onClick={() => send("resume")}>+200 步</button>}
          <button className="btn" disabled={busy || !status.task} onClick={() => send("export")} title="导出为 ONNX，可在网页 / C++ / 手机上推理">导出 ONNX</button>
        </div>
        <BackendSettings conn={conn} />
      </header>
      {view === "data" ? (
        <main className="data-main">
          <DataView onTrain={trainOnData} customRunning={status.task === "custom"} log={log} />
        </main>
      ) : (<>
      {task && <div className="desc">{task.description}</div>}

      <main className="layout">
        <section className="left">
          <div className="stats">
            <div className="stat"><b>{STATE_CN[st] || st}</b><span>状态 · {status.device || "—"}</span></div>
            <div className="stat"><b>{status.step}<small>/{status.total}</small></b><span>迭代步</span><div className="prog"><i style={{ width: pct + "%" }} /></div></div>
            <div className="stat"><b>{fmt(last?.loss)}</b><span>当前 loss</span></div>
            <div className="stat"><b>{last?.acc != null ? (last.acc * 100).toFixed(1) + "%" : "—"}</b><span>准确率</span></div>
            <div className="stat"><b>{status.ips || 0}</b><span>迭代/秒</span></div>
          </div>
          <NetworkGraph graph={graph} phase={pulse} onJump={jump} taskTitle={tasks.find((t) => t.id === status.task)?.title} />
          <Pipeline steps={pipeline} active={activeStage} onJump={jump} source={stageSource} />
          <LineChart title="损失 Loss" dataRef={metricsRef} version={mv} series={LOSS_SERIES} logScale />
          <LineChart title="准确率 · 梯度范数" dataRef={metricsRef} version={mv} series={AUX_SERIES} height={150} />
          <Preview preview={preview} />
          <div className="card logs">
            <div className="card-head"><span>日志</span></div>
            <div className="log-body">
              {logs.slice().reverse().map((l, i) => <div key={i} className={"log " + l.level}><span className="muted">{l.t}</span> {l.msg}</div>)}
            </div>
          </div>
        </section>

        <section className="right">
          <TracePlayer trace={trace} order={order} pos={pos} setPos={(p) => { setPlaying(false); setPos(p); }}
            playing={playing} setPlaying={setPlaying} speed={speed} setSpeed={setSpeed} quiet={quiet} setQuiet={setQuiet}
            onRecord={() => send("trace")} busy={busy} />
          <div className="code-explain">
            <CodePanel files={files} active={activeFile} onSelect={(f) => { setFollow(false); setActiveFile(f); }}
              source={sources[activeFile]} curLine={ev && evFile === activeFile ? ev.l : null}
              visited={visited[activeFile]} showComments={showComments} setShowComments={setShowComments}
              pin={pin && pin.file === activeFile ? pin : null} />
            {!follow && trace && <button className="chip follow" onClick={() => setFollow(true)}>跟随执行位置</button>}
            <div className="bottom-pane">
              <div className="bottom-tabs">
                <button className={"mtab" + (bottom === "memory" ? " on" : "")} onClick={() => setBottom("memory")}>🧠 机器记忆库</button>
                <button className={"mtab" + (bottom === "explain" ? " on" : "")} onClick={() => setBottom("explain")}>逐行讲解{trace ? " ●" : ""}</button>
              </div>
              {bottom === "memory"
                ? <MemoryPanel memOpts={memOpts} setMemOpts={setMemOpts} task={taskId} isCustom={taskId === "custom"}
                    memTick={memTick} running={running} log={log} />
                : <Explain trace={trace} ev={ev} path={ev ? paths[order[pos]] : []} source={evFile ? sources[evFile] : null} />}
            </div>
          </div>
        </section>
      </main>
      </>)}
    </div>
  );
}
