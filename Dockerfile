FROM python:3.11-slim

WORKDIR /app

# 系统依赖（部分解析/编译需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 索引与上传目录
RUN mkdir -p data/uploads data/index

# CloudBase 云托管要求监听 80 端口
ENV PORT=80
EXPOSE 80

# 启动时若已有预置文档则自动入库（避免冷启动空库）
CMD ["sh", "-c", "python scripts/seed.py 2>/dev/null; uvicorn src.api.main:app --host 0.0.0.0 --port 80"]
