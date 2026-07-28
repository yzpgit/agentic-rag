# Agentic RAG · 智能研究助手

> 配置驱动的可插拔 RAG 系统 · 原生 HTML 前端 · Agentic + 混合检索 · 双轨评测

## 特性

- **配置驱动可插拔**：检索器 / Reranker / Chunker / Agent / LLM 均可通过 `config/config.yaml` 热切换，新增模块零侵入主流程（工厂模式 + 注册中心）
- **混合检索**：向量(FAISS) + BM25 双路召回 + RRF 融合
- **Agentic 工作流**（阶段2）：LangGraph 意图路由 + 反思重检索 + HITL 澄清
- **Parent-Child Chunking**（阶段2）：小 chunk 检索 + 大 chunk 上下文
- **双轨评测**（阶段4）：确定性指标(nDCG/MRR/Recall@K) + Ragas(faithfulness)
- **原生 HTML 前端**：单文件，SSE 流式，引用溯源，配置热重载

## 当前阶段

阶段1（已完成）：可插拔骨架 + 朴素/混合 RAG 跑通 + 原生 HTML + SSE

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 LLM（OpenAI 兼容）
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 3. 启动服务
uvicorn src.api.main:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

## 使用

- **Web**：浏览器打开后，左侧上传文档 → 对话区提问，右侧查看引用来源
- **CLI 入库**：`python scripts/ingest.py data/docs/`

## 配置切换示例

编辑 `config/config.yaml`（或前端配置面板）：

```yaml
retriever:
  mode: hybrid              # vector / bm25 / hybrid
  vector:
    embedding: bge-m3       # 切换更强 embedding（需装 FlagEmbedding）
llm:
  provider: openai
  model: gpt-4o-mini
```

通过 `OPENAI_BASE_URL` 环境变量可无缝切换到通义千问 / Ollama 等 OpenAI 兼容接口。

## 项目结构

```
agentic-rag/
├── config/config.yaml        # 统一配置入口
├── src/
│   ├── registry.py           # 插件注册中心
│   ├── base.py               # 抽象基类 + 数据结构
│   ├── orchestrator.py       # Pipeline 组装与调度
│   ├── document/             # 解析器（pdf/md/html/docx）
│   ├── chunking/             # 切分器（recursive，阶段2+ parent_child）
│   ├── indexing/             # Embedding 封装
│   ├── retriever/            # 检索器（vector/bm25/hybrid）
│   ├── reranker/             # 精排（阶段2）
│   ├── agent/                # LangGraph 工作流（阶段2）
│   ├── llm/                  # LLM Provider
│   ├── eval/                 # 评测（阶段4）
│   └── api/                  # FastAPI + SSE
├── frontend/index.html       # 单文件原生前端
├── scripts/ingest.py         # 批量入库
└── Dockerfile
```

## 路线图

- [x] 阶段1：骨架 + 朴素/混合 RAG + 前端 + SSE
- [ ] 阶段2：LangGraph Agent + Parent-Child + Rerank
- [ ] 阶段3：可插拔点完善 + HITL 澄清
- [ ] 阶段4：双轨评测 + 消融实验
