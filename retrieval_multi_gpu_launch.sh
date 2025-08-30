#!/bin/bash
cd /mnt/nvme_data/Search-R1
# 直接运行，不在脚本中使用conda activate
python search_r1/search/retrieval_with_summarizer_multi_gpu.py
