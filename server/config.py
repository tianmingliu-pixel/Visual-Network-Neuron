"""
部署配置（全部来自环境变量，本机运行时都不用设置）/ deployment settings from environment variables

  NEUROCORE_DATA_DIR         上传的数据、记忆库、导出的模型放在哪（默认 项目/data；云端可设为持久磁盘 /data）
  NEUROCORE_CLOUD=1          云端模式：只能读取上传到服务器的数据（禁止输入服务器本地路径）、限制上传大小
  NEUROCORE_TOKEN=口令        设置后，开始/停止训练、上传、删除记忆等“写操作”都要带口令；没有口令的人只能观看
  NEUROCORE_ALLOWED_ORIGINS  允许哪些网页跨域访问，逗号分隔，例如 https://visual-network-neuron.vercel.app
                             （默认 *：任何网页都能读；写操作仍受口令保护）
  NEUROCORE_MAX_UPLOAD_MB    单个上传文件上限（云端默认 200，本机默认 2048）
  HOST / PORT                监听地址与端口（云平台通常会提供 PORT）
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOUD = os.environ.get("NEUROCORE_CLOUD", "").lower() in ("1", "true", "yes")
DATA_DIR = os.path.abspath(os.environ.get("NEUROCORE_DATA_DIR") or os.path.join(ROOT, "data"))
EXPORT_DIR = os.path.join(DATA_DIR, "exports") if os.environ.get("NEUROCORE_DATA_DIR") else os.path.join(ROOT, "exports")
TOKEN = os.environ.get("NEUROCORE_TOKEN", "").strip()
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("NEUROCORE_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.environ.get("NEUROCORE_MAX_UPLOAD_MB") or (200 if CLOUD else 2048))
