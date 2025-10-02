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
- [SFT Summarizer Usage](#sft-summarizer-usage)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Inference](#inference)
- [Use Your Own Dataset](#use-your-own-dataset)
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

### API Endpoints

SFT Summarizer provides the following API endpoints:

#### 1. Retrieve and Summarize (single query)
```http
POST http://127.0.0.1:8000/retrieve
Content-Type: application/json

{
    "query": "Your query question",
    "topk": 5,
    "return_scores": false
}
```

#### 1b. (Optional) Batch Retrieve and Summarize
```http
POST http://127.0.0.1:8000/batch_retrieve
Content-Type: application/json

{
    "queries": ["Question 1", "Question 2"],
    "topk": 5,
    "return_scores": false
}
```

#### 2. Summarize Only (Optional)
```http
POST http://127.0.0.1:8000/summarize
Content-Type: application/json

{
    "text": "Text content to be summarized"
}
```

### Integration with Training Pipeline

During training, the model uses SFT Summarizer in the following way:

1. **Query Processing**: Model generates search queries
2. **Document Retrieval**: Uses e5 retriever to find relevant documents
3. **Intelligent Summarization**: SFT Summarizer summarizes retrieved documents
4. **Reasoning Generation**: Performs reasoning and answering based on summarized content

## Training Configuration

**Reward:** Exact-match (EM) computed on the final answer.  
**Loss masking:** Only LLM-generated tokens (reasoning, queries, answers) are optimized; retrieved evidence tokens are masked out of the loss.

### Training Script Overview

This project uses the `train_ppo.sh` script for PPO reinforcement learning training with the following main configurations:

#### Basic Configuration
```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Data path
export DATA_DIR='data/nq_hotpotqa_train'

# Base model
export BASE_MODEL='/mnt/nvme_data/Qwen2.5-3B-Base'
# (Optional) For ablation:
# export BASE_MODEL='/mnt/nvme_data/Qwen2.5-3B-Instruct'

# Experiment name
export EXPERIMENT_NAME=nq_hotpotqa-search-r1-ppo-qwen2.5-3b-base-v0.2-summarizer
```

#### Training Parameters Details

| Parameter Category | Parameter Name | Value | Description |
|-------------------|----------------|-------|-------------|
| **Data Config** | `data.train_batch_size` | 512 | Global training batch size |
| | `data.val_batch_size` | 256 | Validation batch size |
| | `data.max_prompt_length` | 4096 | Maximum prompt length |
| | `data.max_response_length` | 500 | Maximum response length |
| | `data.max_start_length` | 2048 | Maximum start sequence length |
| | `data.max_obs_length` | 500 | Maximum observation length |
| | `data.shuffle_train_dataloader` | True | Shuffle training data |
| **Model Config** | `actor_rollout_ref.actor.optim.lr` | 1e-6 | Actor learning rate |
| | `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | 0.285 | Actor learning rate warmup ratio |
| | `critic.optim.lr` | 1e-5 | Critic learning rate |
| | `critic.optim.lr_warmup_steps_ratio` | 0.015 | Critic learning rate warmup ratio |
| | `algorithm.kl_ctrl.kl_coef` | 0.001 | KL divergence control coefficient |
| | `algorithm.adv_estimator` | gae | Advantage estimation method (GAE) |
| | `algorithm.gamma` | 1.0 | GAE discount factor |
| | `algorithm.lam` | 1.0 | GAE lambda parameter |
| **PPO Config** | `actor_rollout_ref.actor.ppo_mini_batch_size` | 256 | PPO optimizer mini-batch size per update |
| | `actor_rollout_ref.actor.ppo_micro_batch_size` | 128 | Per-device micro batch size (actor) |
| | `critic.ppo_micro_batch_size` | 16 | Per-device micro batch size (critic) |
| | `actor_rollout_ref.actor.clip_ratio` | 0.2 | PPO clipping ratio |
| | `actor_rollout_ref.actor.entropy_coeff` | 0.001 | Entropy regularization coefficient |
| | `critic.cliprange_value` | 0.5 | Value function clipping range |
| **Generation Config** | `actor_rollout_ref.rollout.temperature` | 1.0 | Generation temperature (near-deterministic) |
| | `actor_rollout_ref.rollout.top_p` | 1.0 | Top-p sampling (near-deterministic) |
| | `actor_rollout_ref.rollout.n_agent` | 1 | Number of agents |
| | `max_turns` | 5 | Maximum conversation turns (vs baseline 3) |

> We match Search-R1's decoding recipe (temperature=1.0, top-p=1.0) for apples-to-apples comparison; despite not being greedy (temp=0), this behaves stably in our setup.
| **Memory Config** | `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.8 | GPU memory utilization |
| | `actor_rollout_ref.model.enable_gradient_checkpointing` | true | Enable gradient checkpointing |
| | `actor_rollout_ref.model.use_remove_padding` | True | Remove padding optimization |
| **Optimizer Config** | `actor_rollout_ref.actor.optim.weight_decay` | 0.01 | Weight decay (AdamW) |
| | `actor_rollout_ref.actor.optim.adam_beta1` | 0.9 | Adam beta1 |
| | `actor_rollout_ref.actor.optim.adam_beta2` | 0.999 | Adam beta2 |
| | `actor_rollout_ref.actor.optim.adam_eps` | 1e-08 | Adam epsilon |
| | `actor_rollout_ref.actor.grad_clip` | 1.0 | Gradient clipping |
| | `critic.grad_clip` | 1.0 | Critic gradient clipping |
| **Training Config** | `trainer.total_epochs` | 15 | Total training epochs |
| | `trainer.total_training_steps` | 1005 | Total training steps (≈67 steps/epoch) |
| | `trainer.save_freq` | 100 | Model save frequency |
| | `trainer.test_freq` | 100 | Test frequency |
| | `trainer.n_gpus_per_node` | 4 | GPUs per node |
| | `trainer.nnodes` | 1 | Number of nodes |
| | `trainer.critic_warmup` | 0 | Critic warmup steps |
| **Retrieval Config** | `retriever.url` | "http://127.0.0.1:8000/retrieve" | Retrieval service address |
| | `retriever.topk` | 5 | Number of retrieved documents (ours; baseline uses top-3) |

> Note: The original Search-R1 baseline uses `topk=3` and `max_turns=3`; RECON uses `topk=5` and `max_turns=5`.
| **Precision Config** | `actor_rollout_ref.model.torch_dtype` | bfloat16 | Training precision |
| | `actor_rollout_ref.rollout.torch_dtype` | bfloat16 | Inference precision |
| **Random Seed** | `actor_rollout_ref.actor.megatron.seed` | 1 | Random seed (actor) |
| | `critic.megatron.seed` | 1 | Random seed (critic) |

### Starting Training

#### 1. Ensure Services are Running
```bash
# Launch retrieval server
conda activate retriever
bash retrieval_launch.sh

# Launch SFT Summarizer server
bash retrieval_with_summarizer_launch.sh
```

#### 2. Begin Training
```bash
# Activate training environment
conda activate searchr1

# Start training
bash train_ppo.sh
```

### Training Monitoring

#### Log Files
Training logs are saved to:
```bash
/mnt/nvme_data/$EXPERIMENT_NAME.log
```

#### Model Checkpoints
Model checkpoints are saved to:
```bash
/mnt/nvme_data/verl_checkpoints/$EXPERIMENT_NAME/
```

#### Real-time Monitoring
```bash
# View training logs
tail -f /mnt/nvme_data/$EXPERIMENT_NAME.log

# Check GPU usage
nvidia-smi

# Check training processes
ps aux | grep main_ppo
```

### Custom Training Configuration

#### Modify Model Path
```bash
# Modify in train_ppo.sh
export BASE_MODEL='/your/path/to/model'
```

#### Adjust Training Parameters
```bash
# Modify batch size
data.train_batch_size=256  # Reduce memory usage

# Modify learning rate
actor_rollout_ref.actor.optim.lr=5e-7  # Smaller learning rate

# Modify training steps
trainer.total_training_steps=2000  # More training steps
```

#### Use Different Retrievers
```bash
# Modify retriever URL
retriever.url="http://127.0.0.1:8001/retrieve"  # Different port

# Modify number of retrieved documents
retriever.topk=5  # Ours (baseline uses top-3)
```

### Multi-GPU Training

#### Single Node Multi-GPU
```bash
# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Adjust GPU count
trainer.n_gpus_per_node=8
```

#### Multi-Node Training
```bash
# Set number of nodes
trainer.nnodes=2

# Set GPUs per node
trainer.n_gpus_per_node=4
```

## Results

### 📊 **Accuracy (Exact Match)**

#### **Qwen2.5-3B-Base + PPO**
- **Search-R1**: 0.303 → **RECON**: 0.347 (+14.5%)
- Gains strongest on multi-hop datasets (HotpotQA, 2Wiki, MuSiQue, Bamboogle)

#### **Qwen2.5-7B-Base + PPO**
- **Search-R1**: 0.431 → **RECON**: 0.444

*(Averages computed over the same 7 datasets as in the paper's main table.)*

### ⚡ **Efficiency (Qwen2.5-7B-Base + PPO)**
- **Avg context length**: 948.27 → 619.7 tokens
- **Avg inference time**: 28.79s → 19.9s
- **Avg search turns**: 2.13 → 1.84

*(Averages computed over the same 7 datasets as in the paper's main table.)*

### 🚀 **Training Speed**
- **3B on 4× H200, 500 steps**: RECON 13.9h vs Search-R1 14.7h (≈5.2% faster)

*(Timing measured on a 500-step run; full training runs in the paper use 1005 steps / 15 epochs.)*

### 🧪 **Ablations**
#### **Instruct vs Base (3B)**
- **Search-R1 → RECON (Avg EM)**:
  - **Base**: 0.303 → 0.347
  - **Instruct**: 0.325 → 0.336
- RECON helps both, with larger gains on Base (smaller backbones benefit more from compression)

### 🧩 **Why It Works**
- **Less noise, more signal**: Summaries remove redundancy and keep only task-relevant evidence, improving the signal-to-noise ratio for the policy.
- **Better multi-turn scaling**: Shorter contexts allow more turns and deeper retrieval without hitting length limits or latency cliffs.
- **Grounded reasoning**: Explicit summaries encourage the policy to anchor each step on external evidence, improving reliability over purely generative chains.


## Inference
#### You can play with the trained Search-R1 model with your own question.
(1) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) Run inference.
```bash
conda activate searchr1
python infer.py
```
You can modify the ```question``` on line 7 to something you're interested in.

## Use your own dataset

### QA data
For each question-answer sample, it should be a dictionary containing the desired content as below:

```
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

You can refer to ```scripts/data_process/nq_search.py``` for a concrete data processing example.

### Corpora

It is recommended to make your corpus a jsonl file, where each line (a dictionary with "id" key and "contents" key) corresponds to one passage. You can refer to ```example/corpus.jsonl``` for an example.

The "id" key corresponds to the passage id, while the "contents" key corresponds to the passage content ('"' + title + '"\n' + text).
For example:
```
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 \u2013 July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
...
{"id": "100", "contents": "Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate."}
...
```

**Index your corpora (optional).**
If you would like to use a local retriever as the search engine, you can index your own corpus by:
```
bash search_r1/search/build_index.sh
```
You can change ```retriever_name``` and ```retriever_model``` to your interested off-the-shelf retriever.

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
