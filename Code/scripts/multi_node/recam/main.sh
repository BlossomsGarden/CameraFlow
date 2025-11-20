#!/bin/bash
# 增加HCCL连接超时时间，防止多节点分布式训练时因计算速度差异导致超时
# 默认120秒，这里设置为3600秒（1小时）以应对奖励计算等耗时操作
export HCCL_CONNECT_TIMEOUT=3600


MASTER_PORT=19001
RANK=0
MASTER_ADDR=172.16.0.132


nohup accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_machines 4 \
    --num_processes 32 \
    --machine_rank ${RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    scripts/train_recam.py \
    --config config/grpo.py:my_recam_8npu \
    > train-RANK${RANK}.out 2>&1 &
