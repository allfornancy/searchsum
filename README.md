# Search-R1 with SFT Summarizer: Enhanced Reasoning and Search Training Framework

<div align="center">
  <img src="https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/main/public/logo.png" alt="logo" width="300"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2503.09516">
    <img src="https://img.shields.io/badge/Paper1-blue?style=for-the-badge" alt="Button1"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15117">
    <img src="https://img.shields.io/badge/Paper2-green?style=for-the-badge" alt="Button2"/>
  </a>
  <a href="https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5">
    <img src="https://img.shields.io/badge/Resources-orange?style=for-the-badge" alt="Button3"/>
  </a>
  <a href="https://x.com/BowenJin13/status/1895544294473109889">
    <img src="https://img.shields.io/badge/Tweet-red?style=for-the-badge" alt="Button4"/>
  </a>
  <a href="https://wandb.ai/peterjin/Search-R1-v0.2">
    <img src="https://img.shields.io/badge/Logs-purple?style=for-the-badge" alt="Button5"/>
  </a>
</p>

## Project Overview

This project represents a **significant enhancement** to the original [Search-R1](https://github.com/PeterGriffinJin/Search-R1) framework, introducing a novel **SFT Summarizer** component that revolutionizes how language models process and utilize retrieved information during reasoning tasks.

### 🎯 **Key Innovation: SFT Summarizer Integration**

Our primary contribution is the seamless integration of a **Supervised Fine-Tuned Summarizer** into the Search-R1 pipeline, enabling:

- 📝 **Intelligent Document Processing**: Automatically summarize long retrieved documents before feeding them to the reasoning model
- 🧠 **Enhanced Reasoning Capability**: Models can now focus on key information rather than processing entire documents
- ⚡ **Improved Training Efficiency**: Reduced computational overhead while maintaining reasoning quality
- 🔄 **End-to-End Optimization**: Complete pipeline from retrieval → summarization → reasoning

### 🚀 **Enhanced Features Over Original Search-R1:**

- 📝 **SFT Summarizer**: Our core innovation for intelligent text summarization
- 🔍 **Advanced Retrieval Pipeline**: Enhanced support for multiple search engines
- 🧠 **Optimized Model Support**: Improved compatibility with various LLM models (Llama3, Qwen2.5, etc.)
- ⚡ **Streamlined Training**: Refined reinforcement learning methods (PPO, GRPO, etc.)

**Search-R1** is a reinforcement learning framework designed for training **reasoning-and-searching interleaved LLMs**—language models that learn to reason and make tool calls (e.g., to search engines) in a coordinated manner.

Built upon [veRL](https://github.com/volcengine/verl), Search-R1 extends the ideas of **DeepSeek-R1(-Zero)** by incorporating interleaved search engine access and provides a fully open-source RL training pipeline. It serves as an alternative and open solution to **OpenAI DeepResearch**, enabling research and development in tool-augmented LLM reasoning.

We support different RL methods (e.g., PPO, GRPO, reinforce), different LLMs (e.g., llama3, Qwen2.5, etc) and different search engines (e.g., local sparse/dense retrievers and online search engines).

Original Papers: [link1](https://arxiv.org/pdf/2503.09516), [link2](https://arxiv.org/abs/2505.15117); Models and Data: [link](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5); Twitter Thread: [link](https://x.com/BowenJin13/status/1895544294473109889); Full Experiment Logs: [prelim](https://wandb.ai/peterjin/Search-R1-open); [v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train); [v0.2](https://wandb.ai/peterjin/Search-R1-v0.2); [v0.3](https://wandb.ai/peterjin/Search-R1-v0.3). Details about these logs and methods can be found [here](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md).


![single-turn](public/main.png)

## News

- [2025.7] Search-R1 is supported by [SkyRL](https://github.com/NovaSky-AI/SkyRL)! Detailed instructions: [code](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples/search), [Document](https://novasky-ai.notion.site/skyrl-searchr1).
- [2025.6] Search-R1 is now integrated into the latest version of veRL and can take advantage of its most up-to-date features! Detailed instructions: [veRL](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html), [English Document](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md), [Chinese Document](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md).
- [2025.5] The second [paper](https://arxiv.org/abs/2505.15117) conducting detailed empirical studies is published with logs: [v0.3](https://wandb.ai/peterjin/Search-R1-v0.3). 
- [2025.4] We support [multinode](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/multinode.md) training for 30B+ LLMs!
- [2025.4] We support [different search engines](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md) including sparse local retriever, dense local retriever with ANN indexing and online search engines!
- [2025.3] The first Search-R1 [paper](https://arxiv.org/pdf/2503.09516) is published with the logs: [v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train); [v0.2](https://wandb.ai/peterjin/Search-R1-v0.2).
- [2025.2] We opensource Search-R1 codebase with [preliminary results](https://wandb.ai/peterjin/Search-R1-open).

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

Train a reasoning + search LLM on NQ dataset with e5 as the retriever and Wikipedia as the corpus.

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

**SFT Summarizer is our core contribution and innovation** in this enhanced Search-R1 framework. This component represents a significant advancement over the original Search-R1 by integrating a Supervised Fine-Tuning (SFT) summarizer into the retrieval pipeline. 

**Our Innovation Impact:**
- 🎯 **Novel Architecture**: First to integrate SFT-based summarization into Search-R1's reasoning pipeline
- 📈 **Performance Enhancement**: Dramatically improves model's ability to process long documents efficiently
- 🔧 **Seamless Integration**: Maintains Search-R1's original functionality while adding intelligent summarization
- 🚀 **Production Ready**: Provides RESTful API for easy deployment and integration

This innovation enables intelligent summarization of retrieved documents, significantly improving the model's understanding and processing capabilities for long documents while maintaining the original Search-R1's reasoning quality.

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
| **Data Config** | `data.train_batch_size` | 512 | Training batch size |
| | `data.val_batch_size` | 256 | Validation batch size |
| | `data.max_prompt_length` | 4096 | Maximum prompt length |
| | `data.max_response_length` | 500 | Maximum response length |
| **Model Config** | `actor_rollout_ref.actor.optim.lr` | 1e-6 | Actor learning rate |
| | `critic.optim.lr` | 1e-5 | Critic learning rate |
| | `algorithm.kl_ctrl.kl_coef` | 0.001 | KL divergence control coefficient |
| **Training Config** | `trainer.total_epochs` | 15 | Total training epochs |
| | `trainer.total_training_steps` | 1005 | Total training steps |
| | `trainer.save_freq` | 100 | Model save frequency |
| **Retrieval Config** | `retriever.url` | "http://127.0.0.1:8000/retrieve" | Retrieval service address |
| | `retriever.topk` | 3 | Number of retrieved documents |

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

## Preliminary results

(1) The base model (llama3.2-3b-base) learns to call the search engine and obtain improved performance.

![llama-3b](public/llama32-3b.png)


(2) The base model (Qwen2.5-7b-base) can learn to conduct multi-turn search engine calling and reasoning with RL.

![multi-turn](public/multi-turn.png)

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

### 📝 SFT Summarizer Features (Our Core Innovation & Contribution)
- ✅ **🎯 Novel Architecture**: First implementation of SFT-based summarization in Search-R1 framework
- ✅ **📈 Performance Breakthrough**: Intelligent document summarization with significant efficiency gains
- ✅ **🔧 Seamless Integration**: Perfect integration with retrieval pipeline, maintaining original functionality
- ✅ **🚀 Production-Ready API**: RESTful API for retrieval and summarization services
- ✅ **⚡ GPU Optimization**: GPU-accelerated summarization generation for real-time processing
- ✅ **🎛️ Advanced Configurability**: Support for custom summarization length and style parameters

### 🚀 Training Features
- ✅ **End-to-End Training**: Complete training pipeline from retrieval to summarization to reasoning
- ✅ **Reinforcement Learning**: Uses PPO and other algorithms to optimize search and reasoning capabilities
- ✅ **Multi-turn Dialogue**: Support for multi-turn search and reasoning interactions
- ✅ **Real-time Monitoring**: Complete training logs and checkpoint management

## Our Contributions

### 🎯 **Primary Innovation: SFT Summarizer Integration**

Our main contribution to the Search-R1 ecosystem is the development and integration of a **Supervised Fine-Tuned Summarizer** that significantly enhances the original framework's capabilities:

#### **Technical Contributions:**
- 🔬 **Novel Architecture**: First to implement SFT-based summarization within Search-R1's reasoning pipeline
- 📊 **Performance Optimization**: Reduced computational overhead while maintaining reasoning quality
- 🔧 **Seamless Integration**: Maintained backward compatibility with original Search-R1 functionality
- 🚀 **Production Deployment**: Developed RESTful API for easy integration and deployment

#### **Research Impact:**
- 📈 **Enhanced Efficiency**: Models can now process long documents more efficiently through intelligent summarization
- 🧠 **Improved Reasoning**: Better focus on key information rather than processing entire documents
- 🔄 **End-to-End Pipeline**: Complete optimization from retrieval → summarization → reasoning
- ⚡ **Scalability**: GPU-accelerated processing for real-time applications

#### **Implementation Highlights:**
- **File**: `retrieval_with_summarizer_launch.sh` - Our custom launch script
- **File**: `retrieval_with_summarizer.py` - Core SFT Summarizer implementation
- **Configuration**: Enhanced `train_ppo.sh` with summarizer integration
- **Documentation**: Comprehensive usage guides and API documentation

### 🏆 **Impact on Search-R1 Community**

Our enhancement makes Search-R1 more practical for real-world applications by:
- Reducing computational costs for long document processing
- Improving reasoning quality through focused information extraction
- Providing production-ready deployment options
- Maintaining the original framework's research capabilities

## Acknowledge

The concept of Search-R1 is inspired by [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero/tree/main).
Its implementation is built upon [veRL](https://github.com/volcengine/verl) and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main). 
We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.

## Awesome work powered or inspired by Search-R1

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling Deep Research via Reinforcement Learning in Real-world Environments. [![[code]](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)
- [Multimodal-Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1): Incentivizing LMMs to Search. [![[code]](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [OTC](https://arxiv.org/pdf/2504.14870): Optimal Tool Calls via Reinforcement Learning.
- [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch): Incentivize the Search Capability of LLMs without Searching. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch)](https://github.com/Alibaba-NLP/ZeroSearch)
- [IKEA](https://github.com/hzy312/knowledge-r1): Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent. [![[code]](https://img.shields.io/github/stars/hzy312/knowledge-r1)](https://github.com/hzy312/knowledge-r1)
- [Scent of Knowledge](https://arxiv.org/abs/2505.09316): Optimizing Search-Enhanced Reasoning with Information Foraging.
- [AutoRefine](https://www.arxiv.org/pdf/2505.11277): Search and Refine During Think. [![[code]](https://img.shields.io/github/stars/syr-cn/AutoRefine)](https://github.com/syr-cn/AutoRefine)
- [O^2-Searcher](https://arxiv.org/pdf/2505.16582): A Searching-based Agent Model for Open-Domain Open-Ended Question Answering. [![[code]](https://img.shields.io/github/stars/Acade-Mate/O2-Searcher)](https://github.com/Acade-Mate/O2-Searcher)
- [MaskSearch](https://arxiv.org/pdf/2505.20285): A Universal Pre-Training Framework to Enhance Agentic Search Capability. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/MaskSearch)](https://github.com/Alibaba-NLP/MaskSearch)
- [VRAG-RL](https://arxiv.org/abs/2505.22019): Vision-Perception-Based RAG for Visually Rich Information Understanding. [![[code]](https://img.shields.io/github/stars/Alibaba-NLP/VRAG)](https://github.com/Alibaba-NLP/VRAG)
- [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668): Training LLMs to Reason with Code via SFT and RL. [![[code]](https://img.shields.io/github/stars/yongchao98/R1-Code-Interpreter)](https://github.com/yongchao98/R1-Code-Interpreter)
- [R-Search](https://arxiv.org/abs/2506.04185): Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/QingFei1/R-Search)](https://github.com/QingFei1/R-Search)
- [StepSearch](https://arxiv.org/pdf/2505.15107): Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization. [![[code]](https://img.shields.io/github/stars/Zillwang/StepSearch)](https://github.com/Zillwang/StepSearch)
- [SimpleTIR](https://simpletir.notion.site/report): Stable End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning. [![[code]](https://img.shields.io/github/stars/ltzheng/SimpleTIR)](https://github.com/ltzheng/SimpleTIR)
- [Router-R1](https://arxiv.org/pdf/2506.09033): Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning. [![[code]](https://img.shields.io/github/stars/ulab-uiuc/Router-R1)](https://github.com/ulab-uiuc/Router-R1)
- [SkyRL](https://skyrl.readthedocs.io/en/latest/): A Modular Full-stack RL Library for LLMs. [![[code]](https://img.shields.io/github/stars/NovaSky-AI/SkyRL)](https://github.com/NovaSky-AI/SkyRL)
- [ASearcher](https://arxiv.org/abs/2508.07976): Large-Scale RL for Search Agents. [![[code]](https://img.shields.io/github/stars/inclusionAI/ASearcher)](https://github.com/inclusionAI/ASearcher)
- [ParallelSearch](https://www.arxiv.org/abs/2508.09303): Decompose Query and Search Sub-queries in Parallel with RL. [![[code]](https://img.shields.io/github/stars/Tree-Shu-Zhao/ParallelSearch)](https://github.com/Tree-Shu-Zhao/ParallelSearch)


## Citations

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

```bibtex
@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```
