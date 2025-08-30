"""
检索服务器（GPU 0） + 调用Summarizer API（GPU 4）
"""

import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import requests

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# 设置使用GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        output = self.model(**inputs, return_dict=True)
        query_emb = pooling(output.pooler_output,
                            output.last_hidden_state,
                            inputs['attention_mask'],
                            self.pooling_method)
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()
        return query_emb

class DenseRetriever:
    def __init__(self, config):
        self.config = config
        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            # 使用GPU 0
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = False
            gpu_res = faiss.StandardGpuResources()
            gpu_res.setTempMemory(1024 * 1024 * 1024)  # 限制临时内存为1GB
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index, co)

        self.corpus = load_corpus(config.corpus_path)
        self.encoder = Encoder(
            model_name = config.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

class Config:
    def __init__(self):
        self.retrieval_method = "e5"
        self.retrieval_topk = 5  # 5个文档
        self.index_path = "/mnt/nvme_data/search_data/e5_Flat.index"
        self.corpus_path = "/mnt/nvme_data/search_data/wiki-18.jsonl"
        self.faiss_gpu = True
        self.retrieval_model_path = "intfloat/e5-base-v2"
        self.retrieval_pooling_method = "mean"
        self.retrieval_query_max_length = 256
        self.retrieval_use_fp16 = True
        self.retrieval_batch_size = 256  # 减小批次大小

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 5
    return_scores: bool = False
    use_summarizer: bool = True
    user_questions: Optional[List[str]] = None

app = FastAPI()

# 初始化检索器
config = Config()
retriever = DenseRetriever(config)
print("Retriever initialized on GPU 0")

def call_summarizer(user_question: str, search_query: str, documents: List[str]) -> str:
    """调用GPU 4上的Summarizer API"""
    try:
        prompt = f"""You are a helpful assistant in a retrieval-augmented question-answering system.

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

User Question: {user_question}

Search Query (used by the retriever): {search_query}

Retrieved Documents:
"""
        for i, doc in enumerate(documents[:5], 1):  # 最多5个文档
            prompt += f"[Doc {i}] {doc}\n"
        
        prompt += "\nPlease write the support context for me. Make it clear, factual, and optimized for the aspect defined above."
        
        # 调用Summarizer API（在GPU 4上）
        response = requests.post(
            "http://localhost:5000/summarize",
            json={
                "instruction": prompt,
                "max_length": 800,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['summary']
        else:
            print(f"Summarizer API error: {response.status_code}")
            return "\n".join([f"Doc {i+1}: {doc[:200]}..." for i, doc in enumerate(documents[:3])])
    
    except Exception as e:
        print(f"Error calling summarizer: {e}")
        return "\n".join([f"Doc {i+1}: {doc[:200]}..." for i, doc in enumerate(documents[:3])])

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    增强版检索端点：检索（GPU 0） + 摘要（GPU 4）
    """
    if not request.topk:
        request.topk = config.retrieval_topk

    # 在GPU 0上执行检索
    results, scores = retriever._batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=True
    )
    
    resp = []
    for i, single_result in enumerate(results):
        # 提取文档内容
        documents_text = []
        for doc in single_result:
            content = doc.get('contents', '')
            title = content.split("\n")[0].strip('"') if "\n" in content else "No title"
            text = "\n".join(content.split("\n")[1:]) if "\n" in content else content
            documents_text.append(f"{title}: {text}")
        
        # 如果需要摘要
        if request.use_summarizer and request.user_questions and i < len(request.user_questions):
            # 调用GPU 4上的Summarizer
            summary = call_summarizer(
                user_question=request.user_questions[i],
                search_query=request.queries[i],
                documents=documents_text
            )
            
            # 返回摘要
            resp.append([{
                'document': {
                    'contents': f'"Summary"\n{summary}'
                },
                'score': 1.0
            }])
        else:
            # 返回原始检索结果
            if request.return_scores:
                combined = []
                for doc, score in zip(single_result, scores[i]):
                    combined.append({"document": doc, "score": score})
                resp.append(combined)
            else:
                resp.append(single_result)
    
    return {"result": resp}

if __name__ == "__main__":
    print("Starting retrieval server on GPU 0, Summarizer API on GPU 4...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
