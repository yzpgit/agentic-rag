#!/usr/bin/env python3
"""
CRUD-RAG + RAGAS 评测脚本
=========================
基于 CRUD-RAG 中文基准数据集（arXiv:2401.17043），用 RAGAS 评估 RAG 管道质量。
- 裁判 LLM：LongCat-2.0（通过 OpenAI 兼容接口，默认关闭 thinking）
- Embedding：讯飞云 xop3qwen8bembedding
- 数据集：CRUD-RAG questanswer_1doc 子集（单文档问答）

用法：
  pip install -r requirements-eval.txt
  python scripts/eval_ragas.py [--sample 20] [--mode vector|bm25|hybrid]
"""
from __future__ import annotations
import os
import sys
import json
import asyncio
import argparse
import random
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser(description="CRUD-RAG + RAGAS 评测")
parser.add_argument("--sample", type=int, default=20,
                    help="抽样条数（默认20，建议10-50）")
parser.add_argument("--mode", type=str, default=None,
                    choices=["vector", "bm25", "hybrid"],
                    help="检索模式（默认用 config.yaml 配置）")
parser.add_argument("--output", type=str, default="data/eval/result.json",
                    help="结果输出路径")
parser.add_argument("--seed", type=int, default=42, help="随机种子")


# ============================================================
# 1. 加载 CRUD-RAG 数据集
# ============================================================
CRUD_REPO = "https://github.com/IAAR-Shanghai/CRUD_RAG.git"
CRUD_LOCAL = Path("/tmp/CRUD_RAG")
CRUD_DATA_FILE = ROOT / "data" / "eval" / "crud_split_merged.json"


def download_crud_rag() -> None:
    """下载 CRUD-RAG 数据集"""
    if not CRUD_LOCAL.exists():
        print("  正在 clone CRUD-RAG 仓库...")
        subprocess.run(
            ["git", "clone", "--depth", "1", CRUD_REPO, str(CRUD_LOCAL)],
            check=True,
        )
    src = CRUD_LOCAL / "data" / "crud_split" / "split_merged.json"
    if not src.exists():
        raise FileNotFoundError(f"CRUD-RAG 数据文件不存在: {src}")
    CRUD_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, CRUD_DATA_FILE)
    print(f"  数据已保存: {CRUD_DATA_FILE}")


def load_crud_rag(sample_n: int, seed: int) -> list[dict]:
    """加载 CRUD-RAG questanswer_1doc 子集并抽样"""
    if not CRUD_DATA_FILE.exists():
        print("[!] 未找到 CRUD-RAG 数据，开始下载...")
        download_crud_rag()

    with open(CRUD_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # questanswer_1doc: 单文档问答，最适合 RAG 评测
    qa_data = data["questanswer_1doc"]
    random.seed(seed)
    samples = random.sample(qa_data, min(sample_n, len(qa_data)))
    print(f"  从 {len(qa_data)} 条中抽样 {len(samples)} 条")
    return samples


# ============================================================
# 2. 跑 RAG 管道，收集 answer + contexts
# ============================================================
async def run_rag(pipeline, samples: list[dict]) -> list[dict]:
    """对每个问题跑 RAG，收集生成答案和检索上下文"""
    from src.base import Document

    # 入库：把所有样本的 news1 文档加入知识库
    print("[2/4] 入库文档...")
    total_chunks = 0
    for s in samples:
        doc = Document(
            id=s["ID"],
            content=s["news1"],
            source=f"crud-rag-{s['ID']}",
            metadata={"crud_id": s["ID"], "event": s.get("event", "")},
        )
        chunks = pipeline.chunker.chunk(doc)
        pipeline.retriever.add(chunks)
        total_chunks += len(chunks)
    print(f"  共 {len(samples)} 篇文档 / {total_chunks} 个片段入库")

    # 逐条查询
    print("[3/4] 跑 RAG 生成...")
    results = []
    for i, s in enumerate(samples):
        q = s["questions"]
        gt = s["answers"]

        # 检索
        retrieved = pipeline.retriever.retrieve(q)
        contexts = [r.chunk.content for r in retrieved]

        # 生成（同步调用，不走流式）
        context_str = "\n\n".join(
            f"[{j+1}] {c}" for j, c in enumerate(contexts)
        )
        answer = pipeline.llm.generate(q, context_str)

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "contexts": contexts,
        })
        print(f"  [{i+1}/{len(samples)}] 完成")

    return results


# ============================================================
# 3. RAGAS 评测
# ============================================================
def run_ragas_eval(rag_results: list[dict]) -> dict:
    """用 RAGAS 评测，LongCat 作为裁判"""
    print("[4/4] RAGAS 评测中（LongCat-2.0 作为裁判，约1-3分钟）...")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # 裁判 LLM：LongCat-2.0（默认不开启 thinking，直接返回 content）
    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model="LongCat-2.0",
        temperature=0,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    ))

    # Embedding：讯飞云（走 OpenAI 兼容接口）
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model="xop3qwen8bembedding",
        openai_api_base=os.getenv("EMBEDDING_BASE_URL"),
        openai_api_key=os.getenv("EMBEDDING_API_KEY", "dummy"),
    ))

    # 组装数据集（RAGAS 标准字段：question/answer/contexts/ground_truth）
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rag_results],
        "answer": [r["answer"] for r in rag_results],
        "contexts": [r["contexts"] for r in rag_results],
        "ground_truth": [r["ground_truth"] for r in rag_results],
    })

    # 评测（4 个核心指标）
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        show_progress=True,
    )

    # 提取分数
    scores = {}
    for metric in ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]:
        val = result[metric]
        scores[metric] = float(val.mean()) if hasattr(val, "mean") else float(val)

    # 明细
    try:
        details = result.to_pandas().to_dict("records")
    except Exception:
        details = []

    return {"scores": scores, "details": details}


# ============================================================
# 4. 输出报告
# ============================================================
METRIC_CN = {
    "faithfulness": "忠实度",
    "answer_relevancy": "回答相关性",
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回率",
}


def print_report(eval_result: dict, output_path: Path) -> None:
    print()
    print("=" * 60)
    print("  CRUD-RAG + RAGAS 评测结果")
    print("=" * 60)
    for metric, score in eval_result["scores"].items():
        cn = METRIC_CN.get(metric, metric)
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {cn:12s} ({metric:20s}) {bar} {score:.3f}")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")


# ============================================================
# 主流程
# ============================================================
async def main():
    args = parser.parse_args()

    print("=" * 60)
    print("  CRUD-RAG + RAGAS 评测")
    print("=" * 60)

    # 1. 加载数据
    print("[1/4] 加载 CRUD-RAG 数据集...")
    samples = load_crud_rag(args.sample, args.seed)

    # 2. 构建 Pipeline
    from src.config import get_config
    from src.orchestrator import Pipeline

    cfg = get_config()
    if args.mode:
        cfg["retriever"]["mode"] = args.mode
        print(f"  检索模式: {args.mode}")
    else:
        print(f"  检索模式: {cfg['retriever']['mode']}（来自 config.yaml）")

    pipeline = Pipeline(cfg)

    # 3. 跑 RAG
    rag_results = await run_rag(pipeline, samples)

    # 4. RAGAS 评测
    eval_result = run_ragas_eval(rag_results)

    # 输出
    print_report(eval_result, ROOT / args.output)


if __name__ == "__main__":
    asyncio.run(main())
