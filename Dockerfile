# 医疗健康 Agent API —— 容器镜像（uvicorn app.api:app，端口 8305）
# ⚠️ 未在本机构建实测（本机无 Docker）：按仓库实际入口 app.api:app / 端口 8305 编写，
#    首次构建请先跑：docker build -t med-service-agent .
#    再验证：docker run -p 8305:8305 med-service-agent 后 curl http://127.0.0.1:8305/health
FROM python:3.11-slim

# 不写 .pyc、不缓冲 stdout（日志实时可见）、pip 不留缓存（镜像更小）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先只拷依赖清单安装：requirements.txt 不变时命中缓存层，改业务代码可秒级重建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再拷全部代码（.dockerignore 已排除 .env / .git / __pycache__ / app/sessions 等，
# 密钥绝不进镜像；运行时密钥由 compose 的 env_file 注入）
COPY . .

# 非 root 运行（安全基线）；sessions 为运行时产物目录，容器内按需自动重建
RUN useradd --create-home appuser \
    && mkdir -p app/sessions \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8305

# 无 OPENAI_API_KEY 时自动进入离线 mock 模式，开箱即跑（见 README）
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8305"]
