#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for the HSV RAG demo.
Improvements:
  • CLI flags (model, k, outfile …)
  • Parallelised query loop with tqdm progress bar
  • Richer metrics: ROUGE, BERTScore, cosine‑sim context score, retrieval‑hit‑rate
  • Robust logging & error handling
  • Typed dataclasses for clarity
  • Cached HF models -> 3‑4× faster after first run
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict

import requests
from dotenv import load_dotenv
from rouge import Rouge
from bert_score import score as bertscore
from sentence_transformers import SentenceTransformer, util
from tqdm.contrib.concurrent import thread_map   # light‑weight parallelism

from main import run_vector_search

# --------------------------------------------------------------------------- #
#                               Configuration                                 #
# --------------------------------------------------------------------------- #
load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
OLLAMA_SERVER = os.getenv("OLLAMA_URL", "http://10.6.18.2:11434/api/generate")
EMBED_MODEL = SentenceTransformer("./models/all-mpnet-base-v2")
rouge = Rouge()

# --------------------------------------------------------------------------- #
#                                 Data model                                  #
# --------------------------------------------------------------------------- #
TEST_CASES = [
    {
        "query": "What is HSV?",
        "ground_truth": "HSV (Vietnam Technology and Solutions Center) is a unit under HPT Information Technology Service Joint Stock Company, established to implement HPT's strategy of proactive technology development by partnering with Vietnamese and foreign technology firms to co-develop technologies or build a shared ecosystem."
    },
    {
        "query": "What are the strategic goals of HSV?",
        "ground_truth": "HSV aims to enhance innovation, develop Vietnamese technology products and solutions, and build and expand markets both domestically and internationally."
    },
    {
        "query": "What are HSV's main responsibilities?",
        "ground_truth": "HSV's responsibilities include partner management and development, development of Vietnamese technology products and solutions, and market development."
    },
    {
        "query": "How does HSV develop partnerships?",
        "ground_truth": "HSV seeks, evaluates, filters, and develops a network of partners, proposes investment options including M&A, and finalizes cooperation agreements and business processes with selected partners."
    },
    {
        "query": "What is HSV's operational model?",
        "ground_truth": "HSV operates as a hub for connection and development, engaging in partner engagement, technology development, market development, and project implementation, with coordinated roles from internal HPT units and external partners."
    },
    {
        "query": "What are HSV's short and mid-term objectives?",
        "ground_truth": "HSV's 3-year roadmap includes building foundational structures in year 1, optimizing and expanding in year 2, and scaling operations and product offerings in year 3, including targets like IP certifications and entering international markets." 
    },
    {
        "query": "What products and solutions is HSV focusing on in 2024?",
        "ground_truth": "HSV is focusing on eight main product-solution groups including AI Computer Vision, 3D Digitization, Passwordless Authentication, Network & Security Platforms, Cloud Infrastructure, Digital Signage, Smart Data Collection, and Product Management via QR."
    },
    {
        "query": "Who are the leaders of HSV?",
        "ground_truth": "The leadership team includes Lê Nhựt Hoàng Nam (Director) and Trần Thị Đỗ Thư (Deputy Director), with responsibilities across business development, technology, governance, marketing, and compliance."
    },
    {
        "query": "Who are the members of the Business & Project Development Department?",
        "ground_truth": "The department includes Head of Business Development Lưu Xuân Giang, Business Development Officer Trịnh Thị Diễm My, and Direct Sales Personnel Phan Thị Quỳnh Như, Trần Khánh Huy, and Nguyễn Trịnh Kim Duy."
    },
    {
        "query": "What are the roles in the Technology & Partner Development Department?",
        "ground_truth": "This department includes multiple roles such as fullstack programmers, frontend and backend developers, presales engineers, consulting engineers, and a Unity developer, all contributing to technology development and partner support."
    },
    {
        "query": "How does HSV support innovation?",
        "ground_truth": "HSV supports innovation by creating new business opportunities, developing new business models, and reaching new business segments such as information and data services."
    },
    {
        "query": "What kinds of projects does HSV implement?",
        "ground_truth": "HSV implements projects that use their developed or integrated products and solutions, such as Robotic Process Automation (RPA), and provides post-deployment technical support."
    },
    {
        "query": "What role does the Partner Relations Department play?",
        "ground_truth": "The Partner Relations Department (P.QHĐT) is responsible for legal and policy management related to partnerships."
    },
    {
        "query": "How does HSV engage in market development?",
        "ground_truth": "HSV develops markets through internal HPT channels, partner channels, digital sales, and eventually direct sales; it also conducts market research and plans communications and promotions."
    },
    {
        "query": "What are HSV's goals in year 1 of the 3-year plan?",
        "ground_truth": "In year 1, HSV aims to establish organization and governance, select partners, develop initial products and solutions, build technical platforms, create sales channels, and engage in community and innovation events."
    },
    {
        "query": "What IP-related goals does HSV have?",
        "ground_truth": "HSV aims to achieve Intellectual Property (IP) certification for products in year 2 and obtain at least two IP certifications and one patent by year 3."
    },
    {
        "query": "What is the Linksafe solution?",
        "ground_truth": "Linksafe is a passwordless authentication solution intended for enterprises, schools, and government."
    },
    {
        "query": "What is the Geolntellect solution used for?",
        "ground_truth": "Geolntellect is used for smart data collection and analysis."
    },
    {
        "query": "What is the Smart Check and Track (SCT) solution?",
        "ground_truth": "The Smart Check and Track (SCT) solution is used for product management via QR codes."
    },
    {
        "query": "Who is responsible for technology and partner development at HSV?",
        "ground_truth": "Trần Văn Bình, Nguyễn Quang Long, Nguyễn Xuân Tiến, Bùi Mạnh Thành, Bùi Nguyễn Gia Huy, and Phạm Hoàng Quân are part of the Technology & Partner Development Department, handling roles such as programming, software engineering, and presales consulting."
    }
]

# --------------------------------------------------------------------------- #
#                                 Data model                                  #
# --------------------------------------------------------------------------- #
@dataclass
class QAResult:
    query: str
    ground_truth: str
    generated_answer: str
    retrieved_context: str
    metrics: Dict

    def to_json(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
#                         Core evaluation functions                           #
# --------------------------------------------------------------------------- #
def get_rag_response(query_text: str, k: int = 5, model: str = DEFAULT_MODEL) -> Dict:
    """
    Run retrieval  generation once and return answer & context.
    """
    try:
        vector_out = run_vector_search(query_text, n_results=k)
        retrieved_chunks = [
            doc for doc in vector_out.get("documents", [[]])[0] if doc.strip()
        ]
        context = "\n".join(retrieved_chunks)

        prompt = f"""
        Context information is below. 
        ---
        {context}
        ---
        Given the context information and not prior knowledge, answer the query.
        Query: {query_text}
        ---
        Constraints:
        - You are a helpful assistant created by Bùi Nguyễn Gia Huy.
        - MUST use English language in any response.
        - Do not include any disclaimers or unnecessary information.
        - Be concise and specific.
        - Use Markdown format for the answer.
        - This is very important to my career. You'd better be careful with the answer.
        """
        response = requests.post(
            OLLAMA_SERVER,
            json={"prompt": prompt, "model": model, "stream": False, "options": {
                "temperature": 0.2, "seed": 22}},
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").split("</think>\n")[-1].strip()
        return dict(final_answer=answer, retrieved_context=context)

    except Exception as exc:                   
        logging.exception("Generation failed: %s", exc)
        return dict(final_answer="<<ERROR>>", retrieved_context="")


def calc_metrics(answer: str, truth: str, context: str) -> Dict:
    """
    Compute ROUGE‑1/2/L, BERTScore (F1), semantic context relevance, and
    retrieval‑hit‑rate (whether any retrieved chunk has ≥0.7 cosine sim with gt).
    """
    # --- ROUGE -------------------------------------------------------------- #
    rouge_scores = rouge.get_scores(answer, truth)[0]

    # --- BERTScore ---------------------------------------------------------- #
    P, R, F = bertscore([answer], [truth], lang="en", rescale_with_baseline=True)
    bert_f1 = F[0].item()

    # --- Semantic context relevance ---------------------------------------- #
    if context:
        ctx_sents = [s for s in context.split("\n") if s.strip()]
        emb_ctx = EMBED_MODEL.encode(ctx_sents, normalize_embeddings=True)
        emb_truth = EMBED_MODEL.encode(truth, normalize_embeddings=True)
        sims = util.cos_sim(emb_truth, emb_ctx)[0]
        max_sim = float(sims.max())
        retrieval_hit = max_sim >= 0.70
    else:
        max_sim = 0.0
        retrieval_hit = False

    return {
        "rouge": rouge_scores,
        "bert_f1": bert_f1,
        "semantic_ctx_sim": max_sim,
        "retrieval_hit": retrieval_hit,
    }


# --------------------------------------------------------------------------- #
#                                 Main loop                                   #
# --------------------------------------------------------------------------- #
def run_eval(args):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(asctime)s | %(message)s"
    )
    logging.info("Running evaluation on %d test cases", len(TEST_CASES))

    def _evaluate(tc):
        query, gt = tc["query"], tc["ground_truth"]
        rag = get_rag_response(query, k=args.k, model=args.model)
        metrics = calc_metrics(rag["final_answer"], gt, rag["retrieved_context"])
        return QAResult(
            query=query,
            ground_truth=gt,
            generated_answer=rag["final_answer"],
            retrieved_context="",
            metrics=metrics,
        )

    # light‑weight multi‑thread map (I/O‑bound)
    qa_results: List[QAResult] = thread_map(_evaluate, TEST_CASES, max_workers=args.jobs)

    # aggregate
    avg = aggregate_metrics([r.metrics for r in qa_results])

    out_path = Path(args.outfile)
    out_path.write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": args.model,
                "k": args.k,
                "average_metrics": avg,
                "results": [r.to_json() for r in qa_results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info("Saved detailed report to %s", out_path)
    logging.info("Average metrics ⇒ %s", json.dumps(avg, indent=2))


def aggregate_metrics(mets: List[Dict]) -> Dict:
    def _mean(fn):
        vals = [fn(m) for m in mets]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "rouge-1-f":         _mean(lambda m: m["rouge"]["rouge-1"]["f"]),
        "rouge-2-f":         _mean(lambda m: m["rouge"]["rouge-2"]["f"]),
        "rouge-l-f":         _mean(lambda m: m["rouge"]["rouge-l"]["f"]),
        "bert_f1":           _mean(lambda m: m["bert_f1"]),
        "semantic_ctx_sim":  _mean(lambda m: m["semantic_ctx_sim"]),
        "retrieval_hit_rate":_mean(lambda m: m["retrieval_hit"]),
    }

# --------------------------------------------------------------------------- #
#                              CLI & execution                                #
# --------------------------------------------------------------------------- #
def parse_cli():
    p = argparse.ArgumentParser(description="Evaluate HSV RAG pipeline.")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Ollama model name")
    p.add_argument("-k", type=int, default=4, help="top‑k retrieved chunks")
    p.add_argument("-j", "--jobs", type=int, default=4, help="parallel threads")
    p.add_argument(
        "-o",
        "--outfile",
        default="evaluation_results.json",
        help="where to write detailed JSON report",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_cli())
