#!/usr/bin/env python3
"""
简单的ChromaDB检索服务
只接受问题描述和topk参数，运行在端口9012
"""

import logging
import json
from pathlib import Path
from typing import Any, Final

from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 数据与模型路径
DATA_ROOT: Final[Path] = Path("/home/harry/Issue-Mem-SWE-agent/data/agentic_exp_data_1220_13wDS_6kGPT/chroma_db_experience")
MODEL_PATH: Final[Path] = Path("/home/harry/Issue-Mem-SWE-agent/models/Qwen3-Embedding-0.6B")
DB_DIR: Final[Path] = DATA_ROOT
COLLECTION_NAME: Final[str] = "experience_knowledge"
EXPERIENCE_JSON_PATH: Final[Path] = DATA_ROOT.parent / "experience_data.json"
MAX_QUERY_CHARS: Final[int] = 8000

# 全局变量
chroma_client = None
collection = None
embedding_function = None
experience_data: dict[str, Any] = {}


def _load_experience_data() -> None:
    """Load experience_data.json for mapping Chroma IDs -> repo/fix_experience."""
    global experience_data
    if not EXPERIENCE_JSON_PATH.exists():
        logger.warning("⚠️ experience_data.json not found at %s; patch fields may be empty", EXPERIENCE_JSON_PATH)
        experience_data = {}
        return
    experience_data = json.loads(EXPERIENCE_JSON_PATH.read_text(encoding="utf-8"))
    logger.info("✅ Loaded %d experience records from %s", len(experience_data), EXPERIENCE_JSON_PATH)


def _infer_repo_from_uid(uid: str) -> str:
    """Infer 'owner/repo' prefix from an ID like '<owner>/<repo><digits>'."""
    if not uid:
        return ""
    i = len(uid)
    while i > 0 and uid[i - 1].isdigit():
        i -= 1
    return uid[:i] if i < len(uid) else uid


def _format_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
    formatted_results: list[dict[str, Any]] = []

    metadatas = (raw_results.get("metadatas") or [[]])[0] or []
    documents = (raw_results.get("documents") or [[]])[0] or []
    ids = (raw_results.get("ids") or [[]])[0] or []
    distances = (raw_results.get("distances") or [[]])[0] or []

    total = max(len(metadatas), len(documents), len(ids), len(distances))
    for idx in range(total):
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
        uid = ids[idx] if idx < len(ids) and isinstance(ids[idx], str) else ""
        entry = experience_data.get(uid) if uid else None

        patch = metadata.get("patch")
        if (not patch) and isinstance(entry, dict):
            patch = entry.get("fix_experience") or entry.get("bug_description") or entry.get("content_preview")
        if not patch:
            patch = metadata.get("content_preview") or metadata.get("chroma:document")
        if (not patch) and idx < len(documents) and isinstance(documents[idx], str):
            patch = documents[idx]

        repo = metadata.get("repo")
        if (not repo) and isinstance(entry, dict):
            repo = entry.get("repo")
        if (not repo) and uid:
            repo = _infer_repo_from_uid(uid)

        pr_number = metadata.get("pr_number")
        if (not pr_number) and isinstance(entry, dict):
            pr_number = entry.get("issue_number") or entry.get("issue_id")

        file_val = metadata.get("file") or metadata.get("source_file")
        score = distances[idx] if idx < len(distances) else 0

        result: dict[str, Any] = {"similarity_score": score}
        if repo:
            result["repo"] = str(repo)
        if file_val:
            result["file"] = str(file_val)
        if pr_number:
            result["pr_number"] = str(pr_number)
        if patch:
            result["patch"] = str(patch)
        formatted_results.append(result)

    return formatted_results

def init_chromadb() -> bool:
    """初始化ChromaDB"""
    global chroma_client, collection, embedding_function
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
        if not DB_DIR.exists():
            raise FileNotFoundError(f"Chroma DB directory not found: {DB_DIR}")
        
        # 创建embedding函数
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(MODEL_PATH),
            device="cpu"
        )
        try:
            embedding_function(["warmup"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️ Embedding warmup failed: %s", exc)
        
        # 连接数据库
        chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

        _load_experience_data()
        try:
            # Warm the full query path (index load + embedding) to avoid the first real request timing out.
            collection.query(query_texts=["warmup"], n_results=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️ Chroma query warmup failed: %s", exc)
        
        logger.info("✅ ChromaDB initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaDB: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "chromadb_connected": collection is not None,
        "experience_data_loaded": len(experience_data) > 0
    })

@app.route('/search', methods=['POST'])
def search_patches():
    """搜索patches - 只接受问题描述和topk"""
    try:
        data = request.get_json(silent=True) or {}
        
        # 只接受问题描述和topk参数
        query = data.get('query', '')
        if not isinstance(query, str):
            query = str(query)
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            query = query[:MAX_QUERY_CHARS]
        topk = data.get('topk', 3)
        
        if not query:
            return jsonify({
                "success": False,
                "error": "query parameter is required"
            }), 400
        
        if not isinstance(topk, int) or topk <= 0 or topk > 10:
            return jsonify({
                "success": False,
                "error": "topk must be an integer between 1 and 10"
            }), 400
        
        # 执行搜索
        results = collection.query(
            query_texts=[query],
            n_results=topk
        )
        
        formatted_results = _format_results(results)
        
        return jsonify({
            "success": True,
            "query": query,
            "topk": topk,
            "results": formatted_results,
            "total_results": len(formatted_results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """根路径"""
    return jsonify({
        "message": "ChromaDB Simple Search Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "search": "/search"
        },
        "parameters": {
            "query": "string - 问题描述",
            "topk": "integer - 返回结果数量 (1-10)"
        }
    })

if __name__ == '__main__':
    # 初始化ChromaDB
    if not init_chromadb():
        logger.error("Failed to initialize ChromaDB. Exiting.")
        exit(1)
    
    # 启动服务
    logger.info("🚀 Starting ChromaDB Simple Search Service on port 9012")
    app.run(host='0.0.0.0', port=9012, debug=False, threaded=True)