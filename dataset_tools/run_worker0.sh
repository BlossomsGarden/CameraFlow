#!/bin/bash
# 服务器1：运行前8个脚本（0-8496），使用显卡0-7


ASCEND_RT_VISIBLE_DEVICES="0" nohup python worker0/gen_metadata_qwen3_0_1062.py >> 1.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="1" nohup python worker0/gen_metadata_qwen3_1062_2124.py >> 2.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="2" nohup python worker0/gen_metadata_qwen3_2124_3186.py >> 3.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="3" nohup python worker0/gen_metadata_qwen3_3186_4248.py >> 4.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="4" nohup python worker0/gen_metadata_qwen3_4248_5310.py >> 5.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="5" nohup python worker0/gen_metadata_qwen3_5310_6372.py >> 6.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="6" nohup python worker0/gen_metadata_qwen3_6372_7434.py >> 7.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="7" nohup python worker0/gen_metadata_qwen3_7434_8496.py >> 8.out 2>&1 &

echo "All 8 jobs started on worker0"

