#!/bin/bash
# 服务器2：运行第9-16个脚本（8496-16992），使用显卡0-7

ASCEND_RT_VISIBLE_DEVICES="0" nohup python worker1/gen_metadata_qwen3_8496_9558.py >> 1.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="1" nohup python worker1/gen_metadata_qwen3_9558_10620.py >> 2.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="2" nohup python worker1/gen_metadata_qwen3_10620_11682.py >> 3.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="3" nohup python worker1/gen_metadata_qwen3_11682_12744.py >> 4.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="4" nohup python worker1/gen_metadata_qwen3_12744_13806.py >> 5.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="5" nohup python worker1/gen_metadata_qwen3_13806_14868.py >> 6.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="6" nohup python worker1/gen_metadata_qwen3_14868_15930.py >> 7.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="7" nohup python worker1/gen_metadata_qwen3_15930_16992.py >> 8.out 2>&1 &

echo "All 8 jobs started on worker1"

