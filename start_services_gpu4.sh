#!/bin/bash

# 启动Summarizer API（后台运行）
echo "Starting Summarizer API on GPU 4..."
cd /mnt/nvme_data/searchsum_project
CUDA_VISIBLE_DEVICES=4 python api_server_single_gpu.py > summarizer.log 2>&1 &
SUMMARIZER_PID=$!
echo "Summarizer PID: $SUMMARIZER_PID"

# 等待Summarizer启动
sleep 10

# 检查Summarizer是否正常
curl -s http://localhost:5000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "Summarizer is running"
else
    echo "Failed to start Summarizer"
    exit 1
fi

# 启动增强版检索服务器（也在GPU 4）
echo "Starting Enhanced Retrieval Server on GPU 4..."
cd /mnt/nvme_data/Search-R1
CUDA_VISIBLE_DEVICES=4 python search_r1/search/retrieval_server_with_summarizer.py

# 如果检索服务器退出，也关闭Summarizer
kill $SUMMARIZER_PID
