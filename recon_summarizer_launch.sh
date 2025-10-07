#!/bin/bash
cd /mnt/nvme_data/RECON
conda activate retriever
CUDA_VISIBLE_DEVICES=4 python search_r1/search/retrieval_with_summarizer.py
