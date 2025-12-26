#!/usr/bin/env python3
"""
简单的ChromaDB检索服务
只接受问题描述和topk参数，运行在端口9012
"""

import logging
from pathlib import Path
from typing import Final

import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 数据与模型路径
DATA_ROOT: Final[Path] = Path("/home/harry/Issue-Mem-SWE-agent/data/agentic_exp_data_1220_13wDS_6kGPT/chroma_db_experience")
MODEL_PATH: Final[Path] = "/home/harry/Issue-Mem-SWE-agent/models/Qwen3-Embedding-0.6B"
DB_DIR: Final[Path] = DATA_ROOT
COLLECTION_NAME: Final[str] = "experience_knowledge"

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

@app.route('/search', methods=['POST'])
def search_patches():
    """搜索patches - 只接受问题描述和topk"""
    try:
        data = request.get_json()
        
        # 只接受问题描述和topk参数
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
        
        # 执行搜索
        results = collection.query(
            query_texts=[query],
            n_results=topk
        )
        
        # 格式化结果
        formatted_results = []
        for i, metadata in enumerate(results['metadatas'][0]):
            result = {
                'patch': metadata.get('patch', ''),
                'file': metadata.get('file', ''),
                'repo': metadata.get('repo', ''),
                'pr_number': metadata.get('pr_number', ''),
                'similarity_score': results['distances'][0][i] if results['distances'] else 0
            }
            formatted_results.append(result)
        
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
    app.run(host='0.0.0.0', port=9012, debug=False)