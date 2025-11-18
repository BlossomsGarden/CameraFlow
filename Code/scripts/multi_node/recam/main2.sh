#!/bin/bash
MASTER_PORT=19001
RANK=2
MASTER_ADDR=172.16.0.132

accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml  --num_machines 4   --num_processes 32    --machine_rank ${RANK}    --main_process_ip ${MASTER_ADDR}    --main_process_port ${MASTER_PORT}     scripts/train_recam.py    --config config/grpo.py:my_recam_8npu
