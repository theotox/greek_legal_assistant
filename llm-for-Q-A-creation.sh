#!/bin/bash
nohup llama-server -m /home/llama/models/huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-q5_k_m.gguf \
--host 0.0.0.0 --port 8080 --alias qwen3.6-mtp \
--ctx-size 32000 \
--n-gpu-layers 99 \
--flash-attn on \
--cache-type-k q8_0 --cache-type-v q8_0 \
--threads 16 --parallel 4 > chat.log 2>&1 &
