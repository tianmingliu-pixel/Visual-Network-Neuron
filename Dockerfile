# NeuroCore 后端（PyTorch 训练 + 实时推送），用于 Hugging Face Spaces / Render / 任何 Docker 主机
# Backend image for Hugging Face Spaces (Docker SDK) or any Docker host. CPU-only PyTorch.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NEUROCORE_CLOUD=1 \
    NEUROCORE_DATA_DIR=/app/data \
    HOST=0.0.0.0 \
    PORT=7860

# Hugging Face Spaces 以 uid 1000 运行 / Spaces run as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p /app/data && chown -R user:user /app

USER user
EXPOSE 7860
CMD ["python", "-m", "server"]
