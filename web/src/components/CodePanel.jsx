// 代码面板：左列代码、右列中文注释；当前执行行高亮并自动滚动；本次迭代执行过的行带标记
// Code panel: code | Chinese comment columns, current line highlighted & auto-scrolled.
import { memo, useEffect, useMemo, useRef } from "react";

const KW = new Set(("def class return if elif else for in while with as import from not and or is None True False " +
  "lambda raise try except finally pass break continue yield assert global nonlocal del async await").split(" "));
const TOKEN = /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|@[\w.]+|\b\d+(?:\.\d+)?(?:e-?\d+)?\b|\b[A-Za-z_]\w*\b|\s+|.)/g;

function highlight(code) {
  const out = [];
  const parts = code.match(TOKEN) || [];
  for (let i = 0; i < parts.length; i++) {
    const t = parts[i];
    let cls = null;
    if (/^["']/.test(t)) cls = "s";
    else if (t[0] === "@") cls = "dec";
    else if (/^\d/.test(t)) cls = "n";
    else if (KW.has(t)) cls = "k";
    else if (t === "self") cls = "self";
    else if (/^[A-Za-z_]\w*$/.test(t)) {
      let j = i + 1; while (j < parts.length && /^\s+$/.test(parts[j])) j++;
      if (parts[j] === "(") cls = /^[A-Z]/.test(t) ? "cls" : "fn";
    }
    out.push(cls ? <span key={i} className={"t-" + cls}>{t}</span> : t);
  }
  return out;
}

const Row = memo(function Row({ ln, cur, pinned, visited, showComments, rowRef, pinRef }) {
  return (
    <div ref={cur ? rowRef : pinned ? pinRef : null}
      className={"row" + (cur ? " cur" : "") + (pinned ? " pin" : "") + (visited ? " visited" : "") + (ln.doc ? " doc" : "")}>
      <span className="ln">{ln.n}</span>
      <code className="code">{ln.doc ? ln.code : highlight(ln.code)}</code>
      {showComments && <span className="cmt">{ln.comment}</span>}
    </div>
  );
});

export default function CodePanel({ files, active, onSelect, source, curLine, visited, showComments, setShowComments, pin }) {
  const rowRef = useRef(null);
  const pinRef = useRef(null);
  const scrollRef = useRef(null);
  useEffect(() => {                                   // 跳转到被点击的代码行 / scroll to a pinned line
    const el = pinRef.current, box = scrollRef.current;
    if (!pin || !el || !box) return;
    const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
    box.scrollTop += r.top - b.top - b.height / 3;
  }, [pin, source]);
  useEffect(() => {
    const el = rowRef.current, box = scrollRef.current;
    if (!el || !box) return;
    const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
    if (r.top < b.top + 30 || r.bottom > b.bottom - 30) {
      box.scrollTop += r.top - b.top - b.height / 2 + r.height / 2;      // 只滚动代码框，不滚动整页
    }
  }, [curLine, active, source]);
  const lines = useMemo(() => source?.lines || [], [source]);
  return (
    <div className="card code-card">
      <div className="card-head tabs">
        <div className="tab-list">
          {files.map((f) => (
            <button key={f} className={"tab" + (f === active ? " on" : "")} onClick={() => onSelect(f)} title={f}>
              {f.split("/").pop()}
            </button>
          ))}
        </div>
        <button className={"chip" + (showComments ? " on" : "")} onClick={() => setShowComments(!showComments)}>中文注释</button>
      </div>
      <div className="code-scroll" ref={scrollRef}>
        {!source ? <div className="empty">加载中…</div> : lines.map((ln) => (
          <Row key={ln.n} ln={ln} cur={ln.n === curLine} pinned={pin?.line === ln.n} visited={visited?.has(ln.n)}
            showComments={showComments} rowRef={rowRef} pinRef={pinRef} />
        ))}
      </div>
    </div>
  );
}
