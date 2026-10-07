"""
NeuroCore 可视化后端 / visualization backend (Starlette + uvicorn)

  GET  /api/tasks            任务列表 / tasks
  GET  /api/source?file=...  源代码（含逐行注释）/ annotated source
  GET  /api/status           当前状态 / status
  GET  /api/stream           SSE 实时推送：status / metrics / preview / trace / log / reset
  POST /api/control          {"cmd": start|pause|resume|stop|trace|export, ...}
  /                          前端页面（web/dist）
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .hub import Hub, safe
from .source import read_source
from .trainer import ROOT, Session

DIST = os.path.join(ROOT, "web", "dist")
API_VERSION = 4          # 数据接口版本：2 = 上传文件夹/元数据表/.npy；3 = 多次上传合并成一个数据集、已上传列表；4 = 机器记忆库


def MemoryStore_call(name):
    """把 MemoryStore 的方法名变成 fn(store, *args) / helper for the memory endpoints."""
    return lambda store, *a: getattr(store, name)(*a)


class NoCacheStatic(StaticFiles):
    """让浏览器每次都检查前端文件是否更新，避免更新代码后还在用旧页面 / always revalidate the UI files."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


# 需要口令的写操作 / endpoints that change state
WRITE_PATHS = ("/api/control", "/api/data/upload", "/api/data/inspect", "/api/data/load",
               "/api/data/evaluate", "/api/memory/action")


class GateMiddleware:
    """跨域（CORS，含 Chrome 访问本机的 Private-Network 预检）+ 写操作口令检查。
    CORS for a separately hosted UI (e.g. Vercel) + token check on write endpoints."""

    def __init__(self, app):
        self.app = app

    def _origin_ok(self, origin):
        from .config import ALLOWED_ORIGINS
        return bool(origin) and ("*" in ALLOWED_ORIGINS or origin.rstrip("/") in ALLOWED_ORIGINS)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from .config import TOKEN
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        origin = headers.get("origin", "")
        cors = []
        if self._origin_ok(origin):
            cors = [(b"access-control-allow-origin", origin.encode("latin-1")), (b"vary", b"Origin"),
                    (b"access-control-expose-headers", b"*")]
        if scope["method"] == "OPTIONS" and headers.get("access-control-request-method"):
            extra = [(b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                     (b"access-control-allow-headers", b"content-type, x-neurocore-token"),
                     (b"access-control-max-age", b"600")]
            if headers.get("access-control-request-private-network") == "true":
                extra.append((b"access-control-allow-private-network", b"true"))
            await send({"type": "http.response.start", "status": 204, "headers": cors + extra})
            await send({"type": "http.response.body", "body": b""})
            return
        path = scope.get("path", "")
        if TOKEN and scope["method"] == "POST" and path in WRITE_PATHS:
            import hmac
            given = headers.get("x-neurocore-token", "")
            if not hmac.compare_digest(given.encode(), TOKEN.encode()):
                body = '{"error": "需要口令：这是只读观看模式。点右上角「后端」输入口令后才能训练/上传/修改。", "auth": true}'.encode()
                await send({"type": "http.response.start", "status": 401,
                            "headers": cors + [(b"content-type", b"application/json; charset=utf-8")]})
                await send({"type": "http.response.body", "body": body})
                return

        async def send_with_cors(msg):
            if msg["type"] == "http.response.start" and cors:
                msg = dict(msg, headers=list(msg.get("headers", [])) + cors)
            await send(msg)
        await self.app(scope, receive, send_with_cors)


def create_app(tasks: dict | None = None) -> Starlette:
    if tasks is None:
        from .tasks import TASKS
        tasks = TASKS
    hub = Hub()
    session = Session(hub, tasks)

    @asynccontextmanager
    async def lifespan(app):
        hub.attach(asyncio.get_running_loop())
        flusher = asyncio.create_task(hub.flusher())
        yield
        flusher.cancel()
        session.stop(wait=True)

    def _pipeline(t):
        if hasattr(t, "PIPELINE"):          # 任务可直接提供（测试用）/ task may supply its own
            return t.PIPELINE
        try:
            from .pipeline import build_pipeline
            return build_pipeline(t)
        except Exception:  # noqa  (测试用的简化任务可能没有这些属性)
            return []

    async def list_tasks(request):
        return JSONResponse([{"id": t.id, "title": t.title, "description": t.description,
                              "files": t.files, "defaults": t.defaults, "pipeline": _pipeline(t)}
                             for t in tasks.values()])

    async def source(request):
        try:
            return JSONResponse(read_source(request.query_params.get("file", "")))
        except PermissionError:
            return JSONResponse({"error": "forbidden"}, status_code=403)
        except FileNotFoundError:
            return JSONResponse({"error": "not found"}, status_code=404)

    async def status(request):
        return JSONResponse(session.status())

    async def stream(request: Request):
        q = hub.subscribe()

        async def gen():
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15)
                        yield msg
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield ": keep-alive\n\n"          # 心跳 / heartbeat
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def control(request: Request):
        body = await request.json()
        cmd = body.get("cmd")
        task = body.get("task")
        defaults = tasks[task].defaults if task in tasks else {}
        steps = int(body.get("steps") or defaults.get("steps", 500))
        lr = float(body.get("lr") or defaults.get("lr", 1e-3))
        bs = int(body.get("batch_size") or defaults.get("batch_size", 32))
        loop = asyncio.get_running_loop()
        try:
            if cmd == "start":
                opts = {k: body.get(k) for k in ("memory", "resume", "replay", "augment", "mem_every", "replay_strength")
                        if body.get(k) is not None}
                await loop.run_in_executor(None, session.start, task, steps, lr, bs, opts)
            elif cmd == "pause":
                session.pause()
            elif cmd == "resume":
                session.resume()
            elif cmd == "stop":
                await loop.run_in_executor(None, session.stop, True)
            elif cmd == "trace":
                await loop.run_in_executor(None, session.request_trace, task, lr, bs)
            elif cmd == "export":
                path = await loop.run_in_executor(None, session.export_onnx)
                return JSONResponse({"ok": True, "path": path, **session.status()})
            else:
                return JSONResponse({"error": f"unknown cmd {cmd}"}, status_code=400)
        except Exception as e:  # noqa
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse(safe({"ok": True, **session.status()}))

    # ---- 数据导入与分析 / data import & analysis ----------------------------------------
    data_state = {"analysis": None}

    async def data_upload(request: Request):
        """原始字节上传（无需 multipart）/ raw-body upload.
        单个文件：?name=文件名
        整个文件夹 / 多个文件：?batch=批次号&rel=相对路径（保留 类别/文件 的子文件夹结构），
        全部传完后用 /api/data/inspect 识别 data/uploads/<批次号>。"""
        from .config import MAX_UPLOAD_MB
        from .datasets import UPLOAD_DIR, extract_zip, safe_name, safe_relpath
        batch = request.query_params.get("batch")
        if batch:
            base = os.path.join(UPLOAD_DIR, safe_name(batch))
            path = os.path.join(base, safe_relpath(request.query_params.get("rel") or request.query_params.get("name", "f")))
        else:
            base = UPLOAD_DIR
            path = os.path.join(UPLOAD_DIR, safe_name(request.query_params.get("name", "upload.bin")))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        size = 0
        try:
            with open(path, "wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1024 ** 2:
                        raise ValueError(f"文件超过 {MAX_UPLOAD_MB} MB 上限")
                    f.write(chunk)
        except Exception as e:  # noqa
            try:
                os.remove(path)
            except OSError:
                pass
            return JSONResponse({"error": f"保存上传文件失败：{e}"}, status_code=400)
        if not batch and path.lower().endswith(".zip"):
            try:
                path = await asyncio.get_running_loop().run_in_executor(None, extract_zip, path)
            except Exception as e:  # noqa
                return JSONResponse({"error": f"解压失败：{e}"}, status_code=400)
        return JSONResponse({"path": base if batch else path, "file": path, "size": size})

    async def data_uploads(request):
        """列出已上传的数据，前端“已上传”面板用 / list uploaded datasets."""
        from .datasets import list_uploads
        return JSONResponse(safe({"items": await asyncio.get_running_loop().run_in_executor(None, list_uploads)}))

    # ---- 机器记忆库 / memory store ------------------------------------------------------
    async def _mem(fn, *a):
        from .memory import get_store
        return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(get_store(), *a))

    async def mem_overview(request):
        from .memory import MemoryStore
        res = await _mem(MemoryStore.overview)
        res["current"] = {"run_id": session.run_id, "dataset": session.dataset_key, "task": session.task_id}
        return JSONResponse(safe(res))

    async def mem_run(request):
        res = await _mem(MemoryStore_call("run_detail"), request.query_params.get("id", ""))
        return JSONResponse(safe(res)) if res else JSONResponse({"error": "没有这次运行"}, status_code=404)

    async def mem_snapshot(request):
        res = await _mem(MemoryStore_call("snapshot_detail"), int(request.query_params.get("id", 0)))
        return JSONResponse(safe(res)) if res else JSONResponse({"error": "没有这个快照"}, status_code=404)

    async def mem_samples(request):
        res = await _mem(MemoryStore_call("sample_memory"), request.query_params.get("dataset", ""))
        return JSONResponse(safe({"items": res}))

    async def mem_resumable(request):
        q = request.query_params
        res = await _mem(MemoryStore_call("resumable"), q.get("task") or None, q.get("dataset") or None)
        return JSONResponse(safe({"items": res}))

    async def mem_action(request):
        body = await request.json()
        act = body.get("action")
        try:
            if act == "query":
                return JSONResponse(safe(await _mem(MemoryStore_call("query"), body.get("sql", ""))))
            if act == "delete_run":
                if body.get("run_id") == session.run_id and session.state in ("running", "paused"):
                    return JSONResponse({"error": "正在训练的这次运行不能删除，请先停止。"}, status_code=400)
                await _mem(MemoryStore_call("delete_run"), body.get("run_id", ""))
                return JSONResponse({"ok": True})
            if act == "forget_samples":
                await _mem(MemoryStore_call("forget_samples"), body.get("dataset", ""))
                return JSONResponse({"ok": True})
        except Exception as e:  # noqa
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
        return JSONResponse({"error": f"unknown action {act}"}, status_code=400)

    async def version(request):
        """前端用来确认后端是新版本（旧版后端没有这个接口）/ lets the UI detect a stale backend."""
        from .config import CLOUD, TOKEN
        given = request.headers.get("x-neurocore-token", "")
        return JSONResponse({"version": API_VERSION, "root": "" if CLOUD else ROOT, "cloud": CLOUD,
                             "auth_required": bool(TOKEN), "authorized": (not TOKEN) or given == TOKEN,
                             "features": ["folder_upload", "metadata_labels", "npy", "xlsx_builtin", "audio_formats", "memory_store"]})

    def _check_path(p):
        """云端模式：只能读取上传到服务器 data/uploads 里的数据 / cloud: uploads only."""
        from .config import CLOUD
        from .datasets import UPLOAD_DIR, DataError
        if CLOUD and p:
            real = os.path.realpath(os.path.expanduser(str(p).strip().strip('"')))
            base = os.path.realpath(UPLOAD_DIR)
            if real != base and not real.startswith(base + os.sep):
                raise DataError("云端版本只能分析上传到服务器的数据，请用「上传文件 / 上传文件夹」。")

    async def data_inspect(request: Request):
        from .datasets import DataError, detect
        body = await request.json()
        try:
            _check_path(body.get("path", ""))
        except DataError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        try:
            return JSONResponse(safe(await asyncio.get_running_loop().run_in_executor(
                None, detect, body.get("path", ""), body.get("target") or None)))
        except DataError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:  # noqa
            return JSONResponse({"error": f"识别失败：{type(e).__name__}: {e}"}, status_code=500)

    async def data_load(request: Request):
        from .datasets import DataError, analyze, load_dataset
        from .tasks import make_custom_task
        body = await request.json()
        try:
            _check_path(body.get("path", ""))
        except DataError as e:
            return JSONResponse({"error": str(e)}, status_code=403)

        def work():
            ds = load_dataset(body.get("path", ""), target=body.get("target") or None,
                              task_type=body.get("task_type") or "auto", img_size=int(body.get("img_size") or 32),
                              max_per_class=int(body.get("max_per_class") or 1000),
                              val_ratio=float(body.get("val_ratio") or 0.2))
            return ds, analyze(ds)
        try:
            ds, analysis = await asyncio.get_running_loop().run_in_executor(None, work)
        except DataError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:  # noqa
            return JSONResponse({"error": f"读取失败：{type(e).__name__}: {e}"}, status_code=500)
        cls = make_custom_task(ds)
        tasks["custom"] = cls
        analysis["task"] = {"id": "custom", "title": cls.title, "description": cls.description}
        data_state["analysis"] = analysis
        return JSONResponse(safe(analysis))

    async def data_current(request):
        return JSONResponse(safe(data_state["analysis"] or {}))

    async def data_evaluate(request):
        t = session.task
        if t is None or getattr(t, "id", "") != "custom":
            return JSONResponse({"error": "请先用导入的数据开始训练（任务：自定义数据）"}, status_code=400)

        def work():
            with session._step_lock:
                return t.evaluate()
        try:
            return JSONResponse(safe(await asyncio.get_running_loop().run_in_executor(None, work)))
        except Exception as e:  # noqa
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    async def no_frontend(request):
        return HTMLResponse("<h2>前端尚未构建 / frontend not built</h2>"
                            "<p>运行 <code>.\\scripts\\run.ps1 -Task build-ui</code> 或在 web/ 下执行 "
                            "<code>npm install && npm run build</code>。开发模式：<code>.\\scripts\\run.ps1 -Task ui-dev</code></p>")

    routes = [
        Route("/api/tasks", list_tasks),
        Route("/api/source", source),
        Route("/api/status", status),
        Route("/api/stream", stream),
        Route("/api/control", control, methods=["POST"]),
        Route("/api/data/upload", data_upload, methods=["POST"]),
        Route("/api/version", version),
        Route("/api/data/uploads", data_uploads),
        Route("/api/memory/overview", mem_overview),
        Route("/api/memory/run", mem_run),
        Route("/api/memory/snapshot", mem_snapshot),
        Route("/api/memory/samples", mem_samples),
        Route("/api/memory/resumable", mem_resumable),
        Route("/api/memory/action", mem_action, methods=["POST"]),
        Route("/api/data/inspect", data_inspect, methods=["POST"]),
        Route("/api/data/load", data_load, methods=["POST"]),
        Route("/api/data/current", data_current),
        Route("/api/data/evaluate", data_evaluate, methods=["POST"]),
    ]
    if os.path.isfile(os.path.join(DIST, "index.html")):
        routes.append(Mount("/", NoCacheStatic(directory=DIST, html=True)))
    else:
        routes.append(Route("/", no_frontend))
    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(GateMiddleware)                # 跨域 + 口令 / CORS + token gate
    app.state.session, app.state.hub = session, hub
    return app
