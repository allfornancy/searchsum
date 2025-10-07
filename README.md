# RECON: Efficient Multi-Hop RAG via Learned Context Compression



## Project Overview

**TL;DR.** We extend the Search-R1 reinforcement-learning framework by inserting a learned summarization module into the reasoning-retrieval loop. Instead of concatenating raw retrieved documents, RECON first condenses evidence into short, clarity-guided summaries and then reasons over the compressed context. This active context compression improves accuracy and efficiency at the same time—especially for multi-hop QA—using only SFT-trained summarizers.

### 🧠 **What is RECON?**

RECON (REasoning with CONdensation) is a drop-in augmentation to Search-R1:
1. At each turn, the agent issues a search query.
2. Instead of appending raw passages, RECON routes the top-k retrieved docs to a summarizer.
3. The summarizer produces a concise, clarity-guided summary, which is appended to the context.
4. The policy model continues reasoning on this compressed, de-noised evidence.

This turns evidence compression into a first-class reasoning tool rather than an offline preprocessing step.



## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [SFT Summarizer Training](#sft-summarizer-training)
- [SFT Summarizer Usage](#sft-summarizer-usage)
- [Inference](#inference)
- [Features](#features)
- [Acknowledgments](#acknowledgments)
- [Citations](#citations)

## Installation

### Search-R1 environment
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

## Quick Start

This guide shows how to train RECON (Search-R1 with SFT Summarizer) on **NQ and HotpotQA datasets**.

### Prerequisites

- **Hardware**: 4× H200/A100 GPUs (or similar)
- **Models**: Qwen2.5-3B-Base (or Qwen2.5-3B-Instruct for ablation)
- **Data**: NQ + HotpotQA training data (already processed in `data/nq_hotpotqa_train/`)

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/allfornancy/searchsum.git
cd searchsum

# Create conda environment
conda create -n searchr1 python=3.9
conda activate searchr1

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -e .

# Install additional requirements
pip install flash-attn --no-build-isolation
pip install wandb
```

### Step 2: Download Corpus and Build Index

```bash
# Set your data path
export DATA_PATH=/path/to/your/data

# Download Wikipedia corpus and build FAISS index
python scripts/download.py --save_path $DATA_PATH
cat $DATA_PATH/part_* > $DATA_PATH/e5_Flat.index
gzip -d $DATA_PATH/wiki-18.jsonl.gz
```

### Step 3: Launch Retrieval Server

```bash
# Update paths in retrieval_launch.sh
file_path=$DATA_PATH
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

# Launch retrieval server
python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 5 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
```

### Step 4: Launch SFT Summarizer

```bash
# Activate retrieval environment
conda activate retriever

# Launch SFT Summarizer server
bash retrieval_with_summarizer_launch.sh
```

### Step 5: Start Training

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BASE_MODEL='/path/to/Qwen2.5-3B-Base'  # or Qwen2.5-3B-Instruct
export EXPERIMENT_NAME=nq_hotpotqa-search-r1-ppo-qwen2.5-3b-base-v0.2-summarizer

# Run training
bash train_ppo.sh
```

## SFT Summarizer Training

### Two-Stage Training Process

RECON's summarizer uses a two-stage SFT training approach:

#### Stage 1: Relevance Pretraining (MS MARCO)
```bash
# Download MS MARCO relevance data
git clone https://huggingface.co/datasets/allfornancy/msmarco-sft
```

#### Stage 2: Multi-Aspect Distillation
```bash
# Download mixed training data (NQ + HotpotQA)
git clone https://huggingface.co/datasets/allfornancy/mixed-sft-data-alpaca-complete
```

### Training Configuration
- **Stage 1**: Train classifier on MS MARCO to separate useful vs non-useful passages
- **Stage 2**: Distill GPT-4o-mini summaries across 6 aspects (clarity, factual correctness, completeness, coverage, coherence, logicality)
- **Final Dataset**: ~1.47M training summaries (~1.0M NQ + ~468k HotpotQA)
- **Training Method**: SFT-only (no RL on summarizer)

## SFT Summarizer Usage

> Deployment note: In our reference setup, the retrieval and summarization services run on the **same machine/GPUs** as training, with `gpu_memory_utilization=0.8` on the policy side (≈20% headroom for the services). You can also host them on separate GPUs or a remote node.

### Overview

**RECON's summarizer is our core contribution and innovation** in this enhanced Search-R1 framework. This component represents a significant advancement over the original Search-R1 by integrating a **Two-Stage Summarizer Training (SFT-only)** into the retrieval pipeline.

### 🧪 **Core Contributions**

#### **1. Active Context Compression for RL-RAG**
- Introduces an explicit summarization step within the reasoning–retrieval loop.
- Keeps multi-turn contexts short and focused while preserving task-critical information.
- Improves interpretability via human-readable, aspect-controlled summaries.

#### **2. Two-Stage Summarizer Training (SFT-only)**
- **Stage 1 — Relevance Pretraining (MS MARCO)**: train a classifier to separate useful vs. non-useful passages (initialize summarizer with the classification head removed afterward).
- **Stage 2 — Multi-Aspect Distillation**: distill teacher summaries (GPT-4o-mini) across NQ and HotpotQA, targeted at six aspects: clarity, factual correctness, completeness, coverage, coherence, logicality.
- After deduplication, this yields ~1.47M training summaries (~1.0M NQ + ~468k HotpotQA).
- **Final summarizer is SFT-only (no RL on the summarizer)**.

#### **3. Integration with Search-R1 (PPO-only)**
- We keep the RL backbone and training recipe, but replace raw concatenation with summarization.
- Deeper retrieval and reasoning become feasible: top-3 → top-5 passages per query; 3 turns → 5 turns maximum.
- Policy optimization uses PPO only (GAE + KL control, no GRPO).

#### **4. Accuracy and Efficiency Gains**
- **Accuracy (EM) improves across 7 QA benchmarks**.
- **Qwen2.5-3B-Base + PPO**: 0.303 → 0.347 (Avg EM, +14.5%).
- **Qwen2.5-7B-Base + PPO**: 0.431 → 0.444 (Avg EM).
- **Efficiency (7B backbone)**:
  - Avg context length: 948.3 → 619.7 tokens.
  - Avg inference time: 28.79s → 19.9s.
  - Avg search turns: 2.13 → 1.84.
- **Training speed**: with Qwen2.5-3B-Base + PPO on 4×H200, 13.9h (RECON, 500 steps) vs 14.7h (Search-R1) → 5.2% faster overall.

#### **5. General and Modular**
- The summarizer is plug-and-play: it slots between retrieval and policy, and can be improved independently.
- The summarizer is aspect-aware; you can reweight or ablate aspects (e.g., clarity vs. coverage) without retraining the policy.

### Configuration

Main configuration parameters for SFT Summarizer:

- **GPU Device**: Default uses `CUDA_VISIBLE_DEVICES=4` (same-machine deployment)
- **Server Port**: Default runs on `http://127.0.0.1:8000`
- **Retriever**: Uses e5-base-v2 model for document retrieval
- **Summarizer**: Integrates SFT-trained summarization model
- **Deployment**: For separate machines/GPUs, adjust `CUDA_VISIBLE_DEVICES` and URL accordingly
- **Memory Management**: When co-locating services with training, we set `gpu_memory_utilization=0.8` for the policy to leave ~20% headroom for the services on the same GPUs

### Integration with Training Pipeline

During training, the model uses SFT Summarizer in the following way:

1. **Query Processing**: Model generates search queries
2. **Document Retrieval**: Uses e5 retriever to find relevant documents
3. **Intelligent Summarization**: SFT Summarizer summarizes retrieved documents
4. **Reasoning Generation**: Performs reasoning and answering based on summarized content

## Inference

#### You can play with the trained RECON model with your own question.

(1) Launch retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) Launch SFT summarizer server.
```bash
bash retrieval_with_summarizer_launch.sh
```

(3) Run RECON inference.
```bash
conda activate searchr1
python infer_with_summarizer.py
```

You can modify the ```question``` on line 7 to something you're interested in.

## Features

### 🔍 Search Engine Support
- ✅ Local sparse retrievers (e.g., BM25)
- ✅ Local dense retrievers (flat indexing and ANN indexing)
- ✅ Online search engines (Google, Bing, Brave, etc.)
- ✅ Off-the-shelf neural rerankers

### 🧠 Model Support
- ✅ Multiple LLM models (Llama3, Qwen2.5, etc.)
- ✅ Multiple reinforcement learning methods supported by the underlying framework (e.g., PPO; GRPO/REINFORCE are framework-level options). **This project's experiments use PPO-only (no GRPO).**
- ✅ Multi-GPU and multi-node training support

### 📝 RECON Summarizer Features (Our Core Innovation & Contribution)
- ✅ **🎯 Novel Architecture**: First implementation of active context compression within RL-RAG loop
- ✅ **📈 Performance Breakthrough**: Two-stage SFT training (MS MARCO relevance → multi-aspect distillation)
- ✅ **🔧 Seamless Integration**: Drop-in augmentation to Search-R1 with PPO-only policy optimization
- ✅ **🚀 Production-Ready API**: RESTful API for retrieval and summarization services
- ✅ **⚡ GPU Optimization**: GPU-accelerated summarization generation for real-time processing
- ✅ **🎛️ Aspect-Aware Control**: Six controlled aspects (clarity, factual correctness, completeness, coverage, coherence, logicality)
- ✅ **🔌 Modular Design**: Plug-and-play summarizer that can be improved independently

### 🚀 Training Features
- ✅ **End-to-End Training**: Complete training pipeline from retrieval to summarization to reasoning
- ✅ **Reinforcement Learning**: Uses PPO and other algorithms to optimize search and reasoning capabilities
- ✅ **Multi-turn Dialogue**: Support for multi-turn search and reasoning interactions
- ✅ **Real-time Monitoring**: Complete training logs and checkpoint management

## Acknowledgments

We sincerely thank the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) team for their groundbreaking work on training LLMs to reason and leverage search engines with reinforcement learning. Their innovative framework provided the foundation for our enhanced version with SFT Summarizer integration.

## Citations

```bibtex
@article{jin2025search,
  title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```
