#!/usr/bin/env python3
"""
ChromaDB 检索服务
提供 /search (语义检索) 和 /get_patch (精确获取) 接口
运行在端口 9012
"""

from flask import Flask, request, jsonify
import chromadb
from chromadb.utils import embedding_functions
import logging
import json
import os

# ================= 服务配置区域 =================
# 向量数据库配置
LOCAL_MODEL_PATH = '/home/harry/Issue-Mem-SWE-agent/data/issue_pr_v0/embedding'
DB_DIR = "/root/autodl-tmp/agentic_issue_db"
# 键值映射文件路径 (这是新增的，请确保 build_key_value_map.py 脚本已运行生成此文件)
PATCH_MAP_PATH = '/root/autodl-tmp/repo_pr_patch_map.json' 
# ===============================================

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量
chroma_client = None
collection = None
embedding_function = None
repo_pr_patch_map = {} # 新增：用于精确查询的内存映射

def load_patch_map():
    """加载 Repo/PR# -> Patch 的键值映射文件"""
    global repo_pr_patch_map
    try:
        if os.path.exists(PATCH_MAP_PATH):
            with open(PATCH_MAP_PATH, 'r', encoding='utf-8') as f:
                repo_pr_patch_map = json.load(f)
            logger.info(f"✅ Loaded {len(repo_pr_patch_map)} patch records from {PATCH_MAP_PATH}")
            return True
        else:
            logger.error(f"❌ Patch map file not found at {PATCH_MAP_PATH}. /get_patch will fail.")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to load patch map: {e}")
        return False

def init_chromadb():
    """初始化ChromaDB"""
    global chroma_client, collection, embedding_function
    
    try:
        # 创建embedding函数
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=LOCAL_MODEL_PATH,
            device="cpu"
        )
        
        # 连接数据库
        chroma_client = chromadb.PersistentClient(path=DB_DIR)
        collection = chroma_client.get_collection(
            name="github_pr_patch_data",
            embedding_function=embedding_function
        )
        
        logger.info("✅ ChromaDB initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaDB: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "chromadb_connected": collection is not None,
        "patch_map_loaded": len(repo_pr_patch_map) > 0
    })

@app.route('/search', methods=['POST'])
def search_patches():
    """工具1接口：语义搜索（不变）"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        topk = data.get('topk', 3)
        
        if not query:
            return jsonify({"success": False, "error": "query parameter is required"}), 400
        
        # 执行搜索
        results = collection.query(
            query_texts=[query],
            n_results=topk
        )
        
        # 格式化结果 (只返回 Repo, PR Number, Score)
        formatted_results = []
        if results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                result = {
                    # 注意：这里我们不再返回 'patch' 字段以符合新的工具设计
                    'repo': metadata.get('repo', ''),
                    # 确保 pr_number 是字符串，方便客户端处理
                    'pr_number': str(metadata.get('pr_number', '')), 
                    "issue_content": str(metadata.get('issue_content', '')), 
                    "pr_title": str(metadata.get('pr_title', '')), 
                    "pr_content": str(metadata.get('pr_content', '')), 
                    'similarity_score': results['distances'][0][i] if results['distances'] else 0
                }
                formatted_results.append(result)
        
        return jsonify({
            "success": True,
            "results": formatted_results,
            "total_results": len(formatted_results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_patch', methods=['POST'])
def get_patch_by_meta():
    """工具2接口：根据 Repo 和 PR Number 精确获取 Patch (使用内存映射)"""
    try:
        data = request.get_json()
        repo = data.get('repo')
        pr_number = data.get('pr_number')

        if not repo or not pr_number:
            return jsonify({"success": False, "error": "repo and pr_number are required"}), 400

        # 构造键: owner/repo#pr_number
        key = f"{repo}#{pr_number}"
        
        # 从内存映射中快速查询
        patch_content = repo_pr_patch_map.get(key)

        if patch_content is not None:
             # 为了兼容性，我们模拟 ChromaDB 的结果格式，将 patch 放入一个列表
            return jsonify({
                "success": True,
                "results": [{
                    'repo': repo,
                    'pr_number': pr_number,
                    'patch': patch_content
                }],
                "count": 1
            })
        else:
            return jsonify({
                "success": True,
                "results": [],
                "count": 0
            })

    except Exception as e:
        logger.error(f"Get patch error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # 1. 加载 Patch 映射 (新的步骤)
    load_patch_map() 
    
    # 2. 初始化 ChromaDB
    if not init_chromadb():
        exit(1)
    
    # 3. 启动服务
    logger.info("🚀 Starting ChromaDB Service on port 9012")
    app.run(host='0.0.0.0', port=9012, debug=False)