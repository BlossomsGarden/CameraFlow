#!/bin/bash
# 增加HCCL连接超时时间，防止多节点分布式训练时因计算速度差异导致超时
# 默认120秒，这里设置为3600秒（1小时）以应对奖励计算等耗时操作
export HCCL_CONNECT_TIMEOUT=3600



# # 8 NPU - SD3.5-M
# ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_8gpu


# 8 NPU - Wan2.1-T2V-1.3B
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29502 scripts/train_recam.py --config config/grpo.py:my_recam_8npu
