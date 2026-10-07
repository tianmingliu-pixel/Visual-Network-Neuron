// Vercel 构建步骤：前端已经预先构建在 web/dist，这里只把后端地址写进 config.js（无需安装任何依赖）
// Vercel build: the UI is prebuilt in web/dist; just write the backend URL into config.js.
//   在 Vercel 项目 Settings → Environment Variables 里设置 NEUROCORE_API_BASE，例如
//   https://你的用户名-visual-network-neuron.hf.space
import fs from "fs";
const base = (process.env.NEUROCORE_API_BASE || "").trim().replace(/\/+$/, "");
const js = `// 由 scripts/vercel_build.mjs 生成 / generated at deploy time\nwindow.NEUROCORE_API_BASE = ${JSON.stringify(base)};\n`;
fs.writeFileSync("web/dist/config.js", js);
if (!fs.existsSync("web/dist/index.html")) {
  console.error("web/dist/index.html 不存在：请先在本机运行 .\\scripts\\run.ps1 -Task build-ui 再推送");
  process.exit(1);
}
console.log(base ? `后端地址 → ${base}` : "警告：没有设置 NEUROCORE_API_BASE，网页会尝试连接同源后端（打开后可在右上角「后端」里手动填写）");
