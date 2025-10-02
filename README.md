# RECON: Efficient Multi-Hop RAG via Learned Context Compression



## Project Overview

**TL;DR.** We extend the Search-R1 reinforcement-learning framework by inserting a learned summarization module into the reasoning–retrieval loop. Instead of concatenating raw retrieved documents, RECON first condenses evidence into short, clarity-guided summaries and then reasons over the compressed context. This active context compression improves accuracy and efficiency at the same time—especially for multi-hop QA—without adding RL to the summarizer.

### 🔍 **Motivation**

Retrieval-augmented generation (RAG) systems interleave language model reasoning with external search, but current RL-trained agents (e.g., Search-R1) suffer from:
- **Context bloat**: concatenating long, noisy documents inflates the prompt, slows decoding, and increases cost.
- **Multi-turn accumulation**: multi-hop reasoning compounds redundancy as more text is added each turn.
- **Quality degradation**: irrelevant details distract the policy and make decision-making less reliable.

RECON addresses these pain points by compressing retrieved evidence inside the RL loop via a dedicated summarizer. The policy reads only what matters—short, factual, and well-structured summaries—so it reasons faster and more robustly.

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
- [Preliminary Results](#preliminary-results)
- [Inference](#inference)
- [Use Your Own Dataset](#use-your-own-dataset)
- [Use Your Own Search Engine](#use-your-own-search-engine)
- [Features](#features)
- [Our Contributions](#our-contributions)
- [Acknowledgments](#acknowledgments)
- [Citations](#citations)

## Installation

### Search-r1 environment
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

### Retriever environment (optional)
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```


## Quick Start

Train a reasoning + search LLM on **NQ and HotpotQA dataset combination** with e5 as the retriever and Wikipedia as the corpus.

### Environment Setup

(1) Download the indexing and corpus.
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) Process the NQ dataset.
```bash
python scripts/data_process/nq_search.py
```

### Launch Services

(3) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(4) **Launch SFT Summarizer server** (New feature in this project).
```bash
conda activate retriever
bash retrieval_with_summarizer_launch.sh
```

### Start Training

(5) Run RL training (PPO) with Qwen2.5-3B-Instruct.
```bash
conda activate searchr1
bash train_ppo.sh
```

### Service Descriptions

- **Retrieval Server** (`retrieval_launch.sh`): Launches basic retrieval service
- **SFT Summarizer Server** (`retrieval_with_summarizer_launch.sh`): Launches retrieval service integrated with SFT summarizer
- **Training Script** (`train_ppo.sh`): Launches PPO reinforcement learning training

> **Note**: Ensure that both the retrieval server and SFT Summarizer server are running properly before starting training.

## SFT Summarizer Usage

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

### Launching SFT Summarizer

#### Method 1: Using Launch Script (Recommended)
```bash
# Activate retrieval environment
conda activate retriever

# Launch SFT Summarizer server
bash retrieval_with_summarizer_launch.sh
```

#### Method 2: Direct Python Script Execution
```bash
# Activate retrieval environment
conda activate retriever

# Set GPU device (optional, defaults to GPU 4)
export CUDA_VISIBLE_DEVICES=4

# Run SFT Summarizer directly
python search_r1/search/retrieval_with_summarizer.py
```

### Configuration

Main configuration parameters for SFT Summarizer:

- **GPU Device**: Default uses `CUDA_VISIBLE_DEVICES=4`
- **Server Port**: Default runs on `http://127.0.0.1:8000`
- **Retriever**: Uses e5-base-v2 model for document retrieval
- **Summarizer**: Integrates SFT-trained summarization model

### API Endpoints

SFT Summarizer provides the following API endpoints:

#### 1. Retrieve and Summarize
```bash
POST http://127.0.0.1:8000/retrieve
Content-Type: application/json

{
    "query": "Your query question",
    "topk": 3
}
```

#### 2. Summarize Only
```bash
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

### Troubleshooting

#### Common Issues

1. **GPU Memory Insufficient**
   ```bash
   # Reduce batch size or use smaller model
   export CUDA_VISIBLE_DEVICES=0  # Use different GPU
   ```

2. **Port Already in Use**
   ```bash
   # Check port usage
   lsof -i :8000
   
   # Kill process using the port
   kill -9 <PID>
   ```

3. **Model Loading Failed**
   ```bash
   # Check model path and permissions
   ls -la /path/to/model
   ```

## Training Configuration

### Training Script Overview

This project uses the `train_ppo.sh` script for PPO reinforcement learning training with the following main configurations:

#### Basic Configuration
```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Data path
export DATA_DIR='data/nq_hotpotqa_train'

# Base model
export BASE_MODEL='/mnt/nvme_data/Qwen2.5-3B-Instruct'

# Experiment name
export EXPERIMENT_NAME=nq_hotpotqa-search-r1-ppo-qwen2.5-3b-instruct-v0.2-summarizer
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
| | `retriever.topk` | 5 | Number of retrieved documents (vs baseline 3) |
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
retriever.topk=5  # Retrieve more documents
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

### Troubleshooting

#### Common Training Issues

1. **Insufficient Memory**
   ```bash
   # Reduce batch size
   data.train_batch_size=256
   actor_rollout_ref.actor.ppo_micro_batch_size=64
   ```

2. **Slow Training Speed**
   ```bash
   # Enable gradient checkpointing
   actor_rollout_ref.model.enable_gradient_checkpointing=true
   
   # Use XFORMERS backend
   export VLLM_ATTENTION_BACKEND=XFORMERS
   ```

3. **Retrieval Service Connection Failed**
   ```bash
   # Check if service is running
   curl http://127.0.0.1:8000/health
   
   # Check port usage
   lsof -i :8000
```

## Results

### 📊 **Accuracy (Exact Match)**

#### **Qwen2.5-3B-Base + PPO**
- **Search-R1**: 0.303 → **RECON**: 0.347 (+14.5%)
- Gains strongest on multi-hop datasets (HotpotQA, 2Wiki, MuSiQue, Bamboogle)

#### **Qwen2.5-7B-Base + PPO**
- **Search-R1**: 0.431 → **RECON**: 0.444

### ⚡ **Efficiency (Qwen2.5-7B-Base + PPO)**
- **Avg context length**: 948.27 → 619.7 tokens
- **Avg inference time**: 28.79s → 19.9s
- **Avg search turns**: 2.13 → 1.84

### 🚀 **Training Speed**
- **3B on 4× H200, 500 steps**: RECON 13.9h vs Search-R1 14.7h (≈5.2% faster)

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

## Use your own search engine

Our codebase supports local sparse retriever (e.g., BM25), local dense retriever (both flat indexing with GPUs and ANN indexing with CPUs) and online search engine (e.g., Google, Bing, etc). More details can be found [here](https://github.com/PeterGriffinJin/Search-R1/tree/main/docs/retriever.md).

The main philosophy is to launch a local or remote search engine server separately from the main RL training pipeline. 

The LLM can call the search engine by calling the search API (e.g., "http://127.0.0.1:8000/retrieve").

You can refer to ```search_r1/search/retriever_server.py``` for an example of launching a local retriever server.

## Features

### 🔍 Search Engine Support
- ✅ Local sparse retrievers (e.g., BM25)
- ✅ Local dense retrievers (flat indexing and ANN indexing)
- ✅ Online search engines (Google, Bing, Brave, etc.)
- ✅ Off-the-shelf neural rerankers

### 🧠 Model Support
- ✅ Multiple LLM models (Llama3, Qwen2.5, etc.)
- ✅ Multiple reinforcement learning methods (PPO, GRPO, reinforce)
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

## Our Contributions

### 🎯 **Primary Innovation: Active Context Compression for RL-RAG**

Our main contribution to the Search-R1 ecosystem is the development and integration of **RECON (REasoning with CONdensation)** that significantly enhances the original framework's capabilities:

#### **Technical Contributions:**
- 🔬 **Novel Architecture**: First to implement active context compression within the RL reasoning–retrieval loop
- 📊 **Two-Stage Training**: MS MARCO relevance pretraining → multi-aspect distillation (SFT-only, no RL on summarizer)
- 🔧 **Seamless Integration**: Drop-in augmentation maintaining Search-R1's PPO backbone and training recipe
- 🚀 **Production Deployment**: RESTful API for easy integration and deployment

#### **Research Impact:**
- 📈 **Accuracy Gains**: 3B: 0.303 → 0.347 (+14.5%); 7B: 0.431 → 0.444 (Avg EM across 7 datasets)
- ⚡ **Efficiency Gains**: Context 948 → 620 tokens; latency 28.79s → 19.9s; turns 2.13 → 1.84
- 🚀 **Training Speed**: 5.2% faster wall-clock vs Search-R1 despite extra summarization step
- 🔄 **Deeper Retrieval**: top-3 → top-5 passages per query; 3 → 5 turns maximum

#### **Implementation Highlights:**
- **File**: `retrieval_with_summarizer_launch.sh` - Our custom launch script
- **File**: `retrieval_with_summarizer.py` - Core RECON summarizer implementation
- **Configuration**: Enhanced `train_ppo.sh` with summarizer integration
- **Documentation**: Comprehensive usage guides and API documentation

### 🏆 **Impact on Search-R1 Community**

Our enhancement makes Search-R1 more practical for real-world applications by:
- **Solving context bloat**: Compressing noisy documents to improve signal-to-noise ratio
- **Enabling multi-turn scaling**: Shorter contexts allow more turns without hitting length limits
- **Providing grounded reasoning**: Explicit summaries anchor policy decisions on external evidence
- **Maintaining modularity**: Plug-and-play summarizer can be improved independently

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
