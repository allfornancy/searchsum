#!/usr/bin/env python3

# 读取文件
with open('search_r1/llm_agent/generation.py', 'r') as f:
    content = f.read()

# 查找_batch_search方法并确保有return语句
import re

# 找到_batch_search方法
pattern = r'(def _batch_search\(self, queries, use_summarizer=True\):.*?)\n\n'
match = re.search(pattern, content, re.DOTALL)

if match:
    method_content = match.group(1)
    # 检查是否有return语句
    if 'return requests.post' not in method_content:
        # 需要添加return语句
        # 找到payload定义后添加return
        new_method = method_content.rstrip() + '\n        \n        return requests.post(self.config.search_url, json=payload).json()\n'
        content = content.replace(method_content, new_method)
        
        with open('search_r1/llm_agent/generation.py', 'w') as f:
            f.write(content)
        print("✅ Fixed: Added return statement to _batch_search")
    else:
        print("✅ Method already has return statement")
else:
    print("⚠️ Could not find _batch_search method, please fix manually")

# 验证修复
with open('search_r1/llm_agent/generation.py', 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'def _batch_search' in line:
            # 打印接下来的15行
            for j in range(i, min(i+15, len(lines))):
                print(lines[j], end='')
            break
