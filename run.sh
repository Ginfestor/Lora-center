#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv/bin/activate
python main.py 