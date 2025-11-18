#!/bin/bash

# scp -P 2222 -r /home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO/scripts/accelerate_configs/multi_node.yaml  ma-user@ma-job-1c7c2e94-ba8b-46a9-937a-7148384cc448-worker-1.ma-job-1c7c2e94-ba8b-46a9-937a-7148384cc448:/home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO/scripts/accelerate_configs/

MASTER_PORT=19001
RANK=0
MASTER_ADDR=172.16.0.132

accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml  --num_machines 4   --num_processes 32    --machine_rank ${RANK}    --main_process_ip ${MASTER_ADDR}    --main_process_port ${MASTER_PORT}     scripts/train_recam.py    --config config/grpo.py:my_recam_8npu
