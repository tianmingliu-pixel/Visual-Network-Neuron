// 前端运行时配置 / runtime config for the UI
// 后端地址：留空 = 与页面同一个地址（本机 python -m server 打开时）。
// 部署到 Vercel 时，scripts/vercel_build.mjs 会按环境变量 NEUROCORE_API_BASE 重写这个文件。
window.NEUROCORE_API_BASE = "";
