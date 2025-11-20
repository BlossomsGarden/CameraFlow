#!/bin/bash
# Common part for all nodes
export NCCL_IB_DISABLE=0      # 启用InfiniBand通信（高性能网络）
export NCCL_IB_HCA=mlx5       # 指定使用Mellanox mlx5 InfiniBand网卡
export NCCL_DEBUG=WARN        # 设置NCCL调试信息级别为WARN（只显示警告）
export NCCL_IB_GID_INDEX=3    # 指定InfiniBand GID索引为3


MASTER_PORT=19001             # 主节点通信端口
RANK=0                        # 当前节点排名（主节点为0）
MASTER_ADDR=10.82.139.22      # 主节点IP地址
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_machines 4 \                      # 总节点数：4个
    --num_processes 32 \                    # 总进程数：32个(这意味着每个节点有8张GPU，而不是4张)
    --machine_rank ${RANK} \                # 当前机器排名
    --main_process_ip ${MASTER_ADDR} \      # 主节点IP
    --main_process_port ${MASTER_PORT} \    # 主节点端口
    scripts/train_sd3.py \                  # 训练脚本
    --config config/grpo.py:geneval_sd3     # 配置文件
