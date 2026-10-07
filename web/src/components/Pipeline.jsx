// 训练全过程：数据 → 前向 → 损失 → 反向 → 更新；每一步的含义、公式和对应代码行（点击跳转）
// The five training stages, each linked to the exact source lines.
export default function Pipeline({ steps, active, onJump, source }) {
  if (!steps?.length) return null;
  const cur = steps.find((s) => s.id === active) || null;
  return (
    <div className="card pipeline">
      <div className="card-head"><span>一次训练迭代的全过程</span>
        <span className="muted small-txt">{source === "trace" ? "跟随单步讲解" : source === "anim" ? "训练中（示意循环）" : "点击步骤查看细节"}</span></div>
      <div className="pipe-row">
        {steps.map((s, i) => (
          <div key={s.id} className={"pipe-step" + (s.id === active ? " on" : "")}>
            <div className="pipe-title">{s.title}</div>
            <div className="pipe-formula">{s.formula}</div>
            <div className="pipe-refs">
              {s.refs.map((r, j) => (
                <button key={j} className="ref" onClick={() => onJump(r)} title={r.label}>
                  {r.file.split("/").pop()}:{r.line}
                </button>
              ))}
            </div>
            {i < steps.length - 1 && <span className="pipe-arrow">→</span>}
          </div>
        ))}
      </div>
      <div className="pipe-detail">
        {cur ? <><b>{cur.title}</b> {cur.text}</> : <span className="muted">{steps.map((s) => s.title).join("  ")}：开始训练或点「单步讲解」后，这里会高亮当前所处阶段。</span>}
      </div>
    </div>
  );
}
