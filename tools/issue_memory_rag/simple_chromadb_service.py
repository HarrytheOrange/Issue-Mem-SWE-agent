#!/usr/bin/env python3
"""
ChromaDB 检索服务（经验记忆专用）
只接受问题描述和 topk 参数，返回结构化经验条目，运行在端口 9013。
"""

import logging
from pathlib import Path
from typing import Any, Final

import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 数据与模型路径
MODEL_PATH: Final[Path] = Path("/home/harry/Issue-Mem-SWE-agent/data/issue_pr_v0/embedding")
DB_DIR: Final[Path] = Path("/home/harry/Issue-Mem-SWE-agent/data/issue_memories_verified_top5_v0")
COLLECTION_NAME: Final[str] = "issue_memories_verified_top5_v0"

# 全局变量
chroma_client = None
collection = None
embedding_function = None

def init_chromadb() -> bool:
    """初始化ChromaDB"""
    global chroma_client, collection, embedding_function
    
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
        if not DB_DIR.exists():
            raise FileNotFoundError(f"Chroma DB directory not found: {DB_DIR}")
        
        # 创建embedding函数
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=str(MODEL_PATH),
            device="cpu"
        )
        
        # 连接数据库
        chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
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
        "chromadb_connected": collection is not None
    })

def _format_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
    formatted = []
    metadatas = raw_results.get('metadatas', [[]])[0]
    documents = raw_results.get('documents', [[]])[0]
    distances = raw_results.get('distances', [[]])[0]

    for idx, metadata in enumerate(metadatas):
        formatted.append({
            'source_file': metadata.get('source_file', ''),
            'source_path': metadata.get('source_path', ''),
            'keywords': metadata.get('keywords', ''),
            'description': metadata.get('description', ''),
            'episodic_memory': metadata.get('episodic_memory', ''),
            'semantic_memory': metadata.get('semantic_memory', ''),
            'procedural_memory': metadata.get('procedural_memory', ''),
            'document': documents[idx] if idx < len(documents) else '',
            'similarity_score': distances[idx] if idx < len(distances) else 0,
        })
    return formatted


@app.route('/search', methods=['POST'])
def search_memories():
    """搜索结构化经验"""
    try:
        data = request.get_json()

        query = data.get('query', '')
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
        "message": "ChromaDB Memory Search Service",
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
    logger.info("🚀 Starting ChromaDB Memory Search Service on port 9013")
    app.run(host='0.0.0.0', port=9013, debug=False)