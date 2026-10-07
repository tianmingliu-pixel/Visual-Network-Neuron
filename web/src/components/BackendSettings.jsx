// 右上角「后端」：连接状态 + 选择后端地址（本机 / 云端）+ 写操作口令
// Backend picker: connection state, backend URL (local or cloud) and access token.
import { useEffect, useState } from "react";
import { apiBase, buildBase, getJSON, getToken, resetBackend, saveBackend } from "../api.js";

const LOCAL = "http://127.0.0.1:8765";

export default function BackendSettings({ conn }) {
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState(null);
  const [base, setBase] = useState(apiBase());
  const [token, setToken] = useState(getToken());

  const check = () => getJSON("/api/version").then(setInfo).catch(() => setInfo({ error: true }));
  useEffect(() => { check(); }, [conn]);

  const where = !apiBase() ? "本页面" : apiBase().includes("127.0.0.1") || apiBase().includes("localhost") ? "本机" : "云端";
  const viewOnly = info?.auth_required && !info?.authorized;
  const apply = () => { saveBackend(base.trim(), token.trim()); location.reload(); };

  return (
    <div className="backend">
      <button className={"conn " + conn} onClick={() => setOpen((o) => !o)} title="点击设置后端地址与口令">
        {conn === "open" ? "●" : "○"} 后端 · {where}
        {info?.cloud ? " ☁" : ""}{viewOnly ? " · 只读观看" : ""}
        {conn !== "open" ? (conn === "connecting" ? " · 连接中" : " · 未连接") : ""}
      </button>
      {open && (
        <div className="backend-pop card">
          <div className="card-head"><span>后端连接</span><button className="btn small" onClick={() => setOpen(false)}>✕</button></div>
          <label>后端地址
            <input value={base} onChange={(e) => setBase(e.target.value)} placeholder="留空 = 与本页面同一个地址" />
          </label>
          <div className="quick">
            <button className="chip" onClick={() => setBase("")}>本页面同源</button>
            <button className="chip" onClick={() => setBase(LOCAL)}>本机 {LOCAL}</button>
            {buildBase() && <button className="chip" onClick={() => setBase(buildBase())}>云端（默认）</button>}
          </div>
          <label>口令（只有设置了口令的后端才需要）
            <input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder="NEUROCORE_TOKEN" />
          </label>
          <div className="muted small-txt">
            {info?.error ? "⚠ 连不上这个后端。" : info ? <>
              后端 v{info.version} · {info.cloud ? "云端模式（只能分析上传的数据）" : "本机模式"} ·{" "}
              {info.auth_required ? (info.authorized ? "口令正确，可以训练/上传" : "需要口令：现在是只读观看") : "无需口令"}
            </> : "检查中…"}
          </div>
          <div className="muted small-txt">用本机后端时：先在电脑上运行 <code>.\scripts\run.ps1 -Task ui</code>（训练用你自己的 CPU/GPU，数据不出电脑）。</div>
          <div className="row-end">
            <button className="btn small" onClick={() => { resetBackend(); location.reload(); }}>恢复默认</button>
            <button className="btn small primary" onClick={apply}>保存并重新连接</button>
          </div>
        </div>
      )}
    </div>
  );
}
