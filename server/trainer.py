"""
训练会话：在后台线程中训练，把指标/预览/追踪推给前端
Training session: runs in a background thread and publishes to the Hub.

不卡前端的关键 / why the UI stays smooth:
  1) 训练在独立线程，网页服务器（事件循环）永远空闲可响应
  2) 指标先进缓冲区，由 Hub 每 100ms 合并成一条消息（≤10 次/秒）并抽稀
  3) 预览图按时间节流（≤1 次/秒）
  4) 逐行追踪只在“单步讲解”那一步开启，录完一次性发给前端，在浏览器本地回放
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from typing import Optional

from .hub import Hub
from .tracer import LineTracer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIRS = [os.path.join(ROOT, "neurocore"), os.path.join(ROOT, "server", "tasks.py")]


def pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Session:
    PREVIEW_EVERY_S = 1.0

    def __init__(self, hub: Hub, tasks: dict):
        self.hub, self.tasks = hub, tasks
        self.lock = threading.RLock()
        self.task = None
        self.task_id: Optional[str] = None
        self.state = "idle"            # idle | running | paused | finished | stopped | error
        self.step = 0
        self.total = 0
        self.ips = 0.0
        self.device = None
        self._thread: Optional[threading.Thread] = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._trace_req = threading.Event()
        self._step_lock = threading.Lock()   # 训练步与导出互斥 / serialize train steps vs export
        # 机器记忆库 / memory store
        self.opts = {}                        # memory / replay / augment / mem_every / resume
        self.run_id = None                    # 当前写入的 runs.id
        self.dataset_key = None
        self._prev = None                     # 上一快照的权重副本（算“这段时间权重变了多少”）
        self._last_m = {}                     # 最近一步的训练指标
        self._last_snap = -1

    # ---- 状态 / status -------------------------------------------------------------
    def status(self) -> dict:
        return {"state": self.state, "task": self.task_id, "step": self.step, "total": self.total,
                "device": str(self.device) if self.device else None, "ips": round(self.ips, 1)}

    def _emit_status(self):
        self.hub.publish("status", self.status())

    # ---- 控制命令 / commands --------------------------------------------------------
    def start(self, task_id: str, steps: int, lr: float, batch_size: int, opts: Optional[dict] = None):
        """opts: memory(写入记忆库) resume(从哪个快照继续，或 "best") replay(难例回放) augment(图片增强) mem_every"""
        with self.lock:
            self.stop(wait=True)
            if task_id not in self.tasks:
                raise ValueError(f"unknown task {task_id}")
            self.opts = dict(opts or {}, _steps=int(steps))
            self._create(task_id, lr, batch_size)
            self.hub.publish("reset", {"task": task_id})
            start_step = self._apply_memory(task_id, lr, batch_size, int(steps))
            self.total = start_step + int(steps)
            self._launch(paused=False)

    # ---- 记忆库 / memory ---------------------------------------------------------------
    def _apply_memory(self, task_id, lr, batch_size, steps) -> int:
        """开始训练前：按选项从记忆继续、打开回放/增强、登记一次新的运行。返回起始步数。"""
        import torch
        from .memory import dataset_key, get_store
        o, task = self.opts, self.task
        ds = getattr(task, "dataset", None)
        self.dataset_key = dataset_key(ds) if ds is not None else None
        self.run_id, self._prev, self._last_m, self._last_snap = None, None, {}, -1
        if ds is not None and o.get("augment") and ds.is_image:
            task.augment = True
            self.hub.publish("log", {"msg": "已开启图片数据增强（随机翻转 + 平移）"})
        if not (o.get("memory", True) or o.get("resume")):
            return 0
        store = get_store()
        parent = None
        res = o.get("resume")
        if res:
            if res == "best":
                cands = [c for c in store.resumable(task_id, self.dataset_key) if c["is_best"]]
                if not cands:
                    raise ValueError("记忆库里还没有这个任务/数据集的“最佳”快照，请先完整训练一次。")
                res = cands[0]["id"]
            path = store.ckpt_path(int(res))
            if not path:
                raise ValueError(f"快照 #{res} 的权重文件已不存在（只保留每次运行的“最新”和“最佳”）。")
            snap = store.snapshot_row(int(res))
            ck = torch.load(path, map_location=self.device)
            if ck.get("task") != task_id or (self.dataset_key and ck.get("dataset") != self.dataset_key):
                raise ValueError("这个记忆来自另一个任务或另一份数据，网络形状不同，不能接着训练。")
            task.model.load_state_dict(ck["model"])
            try:
                task.opt.load_state_dict(ck["opt"])
                for g in task.opt.param_groups:          # 用这次界面上设置的学习率 / honor the new lr
                    g["lr"] = float(lr)
            except Exception:  # noqa
                self.hub.publish("log", {"msg": "优化器状态不兼容，只加载了权重", "level": "warn"})
            self.step = int(ck.get("step", snap["step"]))
            parent = int(res)
            va = "—" if snap["val_acc"] is None else "%.1f%%" % (snap["val_acc"] * 100)
            self.hub.publish("log", {"msg": f"🧠 已从记忆快照 #{res}（第 {self.step} 步，验证 {va}）继续训练"})
        if o.get("memory", True):
            n = sum(p.numel() for p in task.model.parameters())
            self.run_id = store.new_run(task_id, task.title, type(task.model).__name__, n,
                                        {"steps": steps, "lr": lr, "batch_size": batch_size, "replay": bool(o.get("replay")),
                                         "augment": bool(task.augment) if ds is not None else False, "start_step": self.step},
                                        self.dataset_key, parent)
            if ds is not None and hasattr(task, "memory_samples"):
                store.register_samples(self.dataset_key, task.memory_samples())
            self._prev = {k: v.detach().float().cpu().clone() for k, v in task.model.state_dict().items()}
            self._last_snap = self.step
            self.hub.publish("memory", {"event": "run", "run_id": self.run_id, "parent": parent})
        if o.get("replay") and ds is not None and hasattr(task, "set_replay"):
            self._refresh_replay(store)
        return self.step

    def _refresh_replay(self, store):
        w = store.replay_weights(self.dataset_key, self.task.tr.tolist(), float(self.opts.get("replay_strength", 2.0)))
        self.task.set_replay(w)
        if w is not None:
            hard = int((w > 1.5).sum())
            self.hub.publish("log", {"msg": f"难例回放：{hard} 个以前容易错的样本被加大抽样权重"})

    def _mem_every(self):
        """每隔多少步存一个快照：默认这次训练步数的 1/40，至少 20 步 / snapshot interval."""
        return int(self.opts.get("mem_every") or max(20, int(self.opts.get("_steps", 800)) // 40))

    def _snapshot(self):
        """把此刻的“机器记忆”写入数据库：指标 + 每层变化 + 样本特征向量 + 检查点。"""
        if not self.run_id or self.step == self._last_snap:
            return
        import torch
        from .memory import get_store
        store, task = get_store(), self.task
        with self._step_lock:
            model = task.model
            sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}   # 必须复制，否则会随训练一起变
            groups = {}
            for name, p in model.named_parameters():
                parts = name.split(".")
                g = ".".join(parts[:2]) if len(parts) > 2 and parts[1].isdigit() else parts[0]
                d = groups.setdefault(g, [0, 0.0, 0.0, 0.0])
                d[0] += p.numel()
                d[1] += float(p.detach().float().pow(2).sum())
                if p.grad is not None:
                    d[2] += float(p.grad.detach().float().pow(2).sum())
                if self._prev is not None and name in self._prev:
                    d[3] += float((sd[name] - self._prev[name]).pow(2).sum())
            layers = [(g, n, w ** 0.5, gg ** 0.5, dd ** 0.5) for g, (n, w, gg, dd) in groups.items()]
            m = dict(self._last_m)
            m["weight_norm"] = sum(w for _, (_, w, _, _) in groups.items()) ** 0.5
            m["delta_norm"] = sum(dd for _, (_, _, _, dd) in groups.items()) ** 0.5 if self._prev is not None else None
            m["lr"] = task.opt.param_groups[0]["lr"]
            embed = None
            if hasattr(task, "memory_embed"):
                embed, vm = task.memory_embed()
                m.update(vm)
            state = {"model": model.state_dict(), "opt": task.opt.state_dict(), "step": self.step,
                     "task": self.task_id, "dataset": self.dataset_key}

            def writer(path):
                torch.save(state, path)
            row = store.add_snapshot(self.run_id, self.step, m, layers, embed, self.dataset_key, writer)
        self._prev = sd
        self._last_snap = self.step
        if self.opts.get("replay") and hasattr(task, "set_replay"):
            self._refresh_replay(store)
        self.hub.publish("memory", {"event": "snapshot", "run_id": self.run_id, "snapshot": row})

    def _create(self, task_id, lr, batch_size):
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))   # 留一个核给网页服务器 / keep a core for the server
        self.device = pick_device()
        self.task = self.tasks[task_id](self.device, float(lr), int(batch_size))
        self.task_id, self.step = task_id, 0
        self.run_id = None                    # 新任务默认不写记忆，start() 里按选项登记 / no memory run until start()
        n = sum(p.numel() for p in self.task.model.parameters())
        self.hub.publish("log", {"msg": f"已创建 {self.task.title}，参数量 {n:,}，设备 {self.device}"})

    def _launch(self, paused: bool):
        self._stop.clear()
        if paused:
            self._pause.set()
        else:
            self._pause.clear()
        self.state = "paused" if paused else "running"
        self._thread = threading.Thread(target=self._run, daemon=True, name="trainer")
        self._thread.start()
        self._emit_status()

    def pause(self):
        if self.state == "running":
            self._pause.set()
            self.state = "paused"
            self._emit_status()

    def resume(self):
        if self.state == "paused":
            self._pause.clear()
            self.state = "running"
            self._emit_status()
        elif self.state in ("finished", "stopped") and self.task is not None:
            self.total = max(self.total, self.step) + 200                 # 继续多训练 200 步 / train 200 more
            self._launch(paused=False)

    def stop(self, wait: bool = False):
        t = self._thread
        if t is not None and t.is_alive():
            self._stop.set()
            self._pause.clear()
            if wait:
                t.join(timeout=10)
            self.state = "stopped"
            self._emit_status()

    def request_trace(self, task_id: Optional[str] = None, lr=1e-3, batch_size=8):
        """请求对下一次迭代做逐行追踪 / trace the next iteration."""
        with self.lock:
            if self.task is None or (task_id and task_id != self.task_id):
                self.stop(wait=True)
                self._create(task_id or "vit", lr, batch_size)
                self.total = self.tasks[self.task_id].defaults["steps"]
                self.hub.publish("reset", {"task": self.task_id})
                self._trace_req.set()
                self._launch(paused=True)        # 线程启动后会先执行追踪步，然后保持暂停
                return
            self._trace_req.set()
            if self._thread is None or not self._thread.is_alive():
                self._launch(paused=True)

    # ---- 训练循环 / loop -------------------------------------------------------------
    def _run(self):
        last_preview = 0.0
        t_window, n_window = time.perf_counter(), 0
        try:
            self._send_preview()
            while not self._stop.is_set():
                if self._trace_req.is_set():                      # 慢动作单步 / traced step
                    self._trace_req.clear()
                    self._traced_step()
                    self._send_preview()
                    continue
                if self._pause.is_set():
                    time.sleep(0.05)
                    continue
                if self.step >= self.total:
                    self.state = "finished"
                    self._send_preview()
                    self._emit_status()
                    self.hub.publish("log", {"msg": f"训练完成：{self.step} 步"})
                    break
                t0 = time.perf_counter()
                with self._step_lock:
                    m = self.task.train_step()                   # 一次普通迭代（不追踪）
                self.step += 1
                n_window += 1
                self._last_m = m
                m["step"] = self.step
                m["dt_ms"] = (time.perf_counter() - t0) * 1000
                self.hub.add_metric(m)
                now = time.perf_counter()
                if now - t_window >= 1.0:
                    self.ips = n_window / (now - t_window)
                    t_window, n_window = now, 0
                    self._emit_status()
                if now - last_preview >= getattr(self.task, "preview_every", self.PREVIEW_EVERY_S):
                    last_preview = now
                    self._send_preview()
                if self.run_id and self.step - self._last_snap >= self._mem_every():
                    self._snapshot()                              # 写入机器记忆 / memory snapshot
            self._finish_memory()
        except Exception as e:  # noqa
            if self.run_id:
                try:
                    from .memory import get_store
                    get_store().set_status(self.run_id, "error", self.step)
                except Exception:  # noqa
                    pass
            self.state = "error"
            self.hub.publish("log", {"msg": f"训练出错: {e}", "level": "error", "trace": traceback.format_exc()})
            self._emit_status()

    def _finish_memory(self):
        if not self.run_id:
            return
        try:
            self._snapshot()
            from .memory import get_store
            get_store().set_status(self.run_id, "finished" if self.step >= self.total else "stopped", self.step)
            self.hub.publish("memory", {"event": "end", "run_id": self.run_id})
        except Exception as e:  # noqa
            self.hub.publish("log", {"msg": f"写入记忆库失败: {e}", "level": "warn"})

    def _traced_step(self):
        import torch
        self.hub.publish("log", {"msg": "正在录制一次完整迭代的逐行轨迹…"})
        tr = LineTracer(TRACE_DIRS)
        t0 = time.perf_counter()
        with self._step_lock, tr:
            m = self.task.train_step()
        # 反向传播与优化器在 C++/PyTorch 内部执行，追加说明事件与梯度统计
        grads = []
        for name, p in self.task.model.named_parameters():
            if p.grad is not None and len(grads) < 40:
                g = p.grad.detach().float()
                grads.append({"k": "tensor", "n": f"∇{name}", "s": list(g.shape), "dt": "float32",
                              "mean": float(g.mean()), "std": float(g.std()) if g.numel() > 1 else 0.0,
                              "min": float(g.min()), "max": float(g.max())})
        tr.add_synthetic("梯度一览（loss.backward() 的结果）",
                         "反向传播由 PyTorch autograd 在 C++ 中完成，按链式法则从损失往回求出每个参数的梯度 ∂loss/∂θ；"
                         "下面列出部分参数的梯度统计（前 40 个）。", grads)
        self.step += 1
        m["step"] = self.step
        m["dt_ms"] = (time.perf_counter() - t0) * 1000
        self.hub.add_metric(m)
        res = tr.result(ROOT)
        res.update(step=self.step, task=self.task_id, loss=m.get("loss"), ms=round(m["dt_ms"], 1))
        self.hub.publish("trace", res)
        self.hub.publish("log", {"msg": f"轨迹已录制：{len(res['events'])} 个事件"
                                        + ("（已截断）" if res["truncated"] else "")})
        if self.state != "running":
            self.state = "paused"
            self._pause.set()
        self._emit_status()

    def _send_preview(self):
        try:
            p = self.task.preview()
            if p is not None:
                p["step"] = self.step
                self.hub.publish("preview", p)
        except Exception as e:  # noqa
            self.hub.publish("log", {"msg": f"预览失败: {e}", "level": "warn"})
        try:
            with self._step_lock:
                g = self.task.graph()                     # 网络结构图 / network graph
            if g is not None:
                g["step"] = self.step
                self.hub.publish("graph", g)
        except Exception as e:  # noqa
            if not getattr(self, "_graph_warned", False):
                self._graph_warned = True
                self.hub.publish("log", {"msg": f"结构图失败: {e}", "level": "warn"})

    # ---- ONNX 导出 / export ------------------------------------------------------------
    def export_onnx(self) -> str:
        if self.task is None:
            raise RuntimeError("请先开始一个训练任务 / start a task first")
        from neurocore.export import export_task_onnx
        with self._step_lock:                     # 等当前这一步结束再导出 / wait for the current step
            from .config import EXPORT_DIR
            return export_task_onnx(self.task, EXPORT_DIR)
