#!/bin/bash
cd /mnt/nvme_data/Search-R1
CUDA_VISIBLE_DEVICES=0 python search_r1/search/retrieval_with_summarizer_gpu0.py
