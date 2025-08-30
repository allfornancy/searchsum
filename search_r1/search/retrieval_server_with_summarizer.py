"""
增强版检索服务器 - 集成Summarizer
"""
import json
import warnings
from typing import List, Optional
import requests
import torch
import numpy as np
from tqdm import tqdm
import datasets
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import logging

# 导入原始检索器代码
from retrieval_server import (
    load_corpus, load_model, pooling, 
    Encoder, BaseRetriever, BM25Retriever, DenseRetriever,
    Config
)

logging.basicConfig(level=logging.INFO)

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 5  # 改为5个文档
    return_scores: bool = False
    use_summarizer: bool = True  # 是否使用摘要器

app = FastAPI()

# Summarizer API配置
SUMMARIZER_URL = "http://localhost:5000/summarize"

def call_summarizer(user_question: str, search_query: str, documents: List[str]) -> str:
    """
    调用Summarizer API生成摘要
    """
    # 使用导师提供的精确prompt模板
    prompt_template = """You are a helpful assistant in a retrieval-augmented question-answering system.
You will be given:
- A user question
- A search prompt (query) used to retrieve information
- A set of documents retrieved using that query
Your task is to generate a support context — a concise, well-structured summary that captures all the key facts from the documents which are relevant to answering the user question.
Important Instructions:
- Do not answer the question directly.
- Do not add external knowledge or hallucinate any content.
- Use only the information found in the retrieved documents.
- Rephrase and compress where appropriate, but preserve factual meaning.
- Maintain consistent tone and structure throughout.
- Focus on maximizing the following aspect in your output:
Focus Aspect: Clarity
→ Write in a clear, accessible manner that is easy to understand. Use simple, direct language and avoid jargon or overly complex sentences. Present information in a straightforward way that makes the key points immediately apparent to the reader. Ensure that each statement is unambiguous and easy to follow.

User Question: {question}

Search Query (used by the retriever): {search_query}

Retrieved Documents:"""
    
    # 构建完整prompt
    instruction = prompt_template.format(
        question=user_question,
        search_query=search_query
    )
    
    # 添加文档
    for i, doc in enumerate(documents, 1):
        instruction += f"\n[Doc {i}] {doc}"
    
    instruction += "\n\nPlease write the support context for me. Make it clear, factual, and optimized for the aspect defined above."
    
    try:
        response = requests.post(
            SUMMARIZER_URL,
            json={
                "instruction": instruction,
                "max_length": 500,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['summary']
        else:
            logging.error(f"Summarizer API error: {response.status_code}")
            # 降级：返回前两个文档的拼接
            return "\n".join(documents[:2])
            
    except Exception as e:
        logging.error(f"Failed to call summarizer: {e}")
        # 降级处理
        return "\n".join(documents[:2])

def format_documents(docs_list):
    """格式化文档内容"""
    formatted_docs = []
    for doc in docs_list:
        if 'contents' in doc:
            content = doc['contents']
        else:
            content = doc.get('text', '')
        
        # 清理格式
        if content:
            title = content.split("\n")[0].strip('"')
            text = "\n".join(content.split("\n")[1:])
            formatted_docs.append(f"{title}: {text}")
    
    return formatted_docs

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    增强版检索端点 - 支持Summarizer
    """
    if not request.topk:
        request.topk = 5  # 默认5个文档
    
    # 执行检索
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=True
    )
    
    # 处理每个查询的结果
    final_results = []
    
    for i, (query, docs) in enumerate(zip(request.queries, results)):
        if request.use_summarizer and len(docs) > 0:
            # 格式化文档
            formatted_docs = format_documents(docs)
            
            # 调用Summarizer生成摘要
            # 注意：这里的user_question和search_query都使用同一个query
            # 在实际应用中，可能需要区分
            summary = call_summarizer(
                user_question=query,
                search_query=query,
                documents=formatted_docs
            )
            
            # 返回摘要作为单个"文档"
            summarized_result = [{
                'document': {
                    'contents': f'"Summary"\n{summary}'
                },
                'score': 1.0  # 摘要的分数设为1
            }]
            
            final_results.append(summarized_result)
        else:
            # 不使用summarizer，返回原始结果
            if request.return_scores:
                combined = []
                for doc, score in zip(docs, scores[i]):
                    combined.append({"document": doc, "score": score})
                final_results.append(combined)
            else:
                final_results.append(docs)
    
    return {"result": final_results}

if __name__ == "__main__":
    # 配置
    config = Config(
        retrieval_method="e5",
        index_path="/mnt/nvme_data/search_data/e5_Flat.index",
        corpus_path="/mnt/nvme_data/search_data/wiki-18.jsonl",
        retrieval_topk=5,  # 检索5个文档
        faiss_gpu=True,
        retrieval_model_path="intfloat/e5-base-v2",
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )
    
    # 加载检索器
    retriever = get_retriever(config)
    logging.info("Retriever loaded successfully")
    
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
