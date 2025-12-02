#!/bin/bash
# 服务器3：运行第17-24个脚本（16992-25488），使用显卡0-7

ASCEND_RT_VISIBLE_DEVICES="0" nohup python worker2/gen_metadata_qwen3_16992_18054.py >> 1.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="1" nohup python worker2/gen_metadata_qwen3_18054_19116.py >> 2.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="2" nohup python worker2/gen_metadata_qwen3_19116_20178.py >> 3.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="3" nohup python worker2/gen_metadata_qwen3_20178_21240.py >> 4.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="4" nohup python worker2/gen_metadata_qwen3_21240_22302.py >> 5.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="5" nohup python worker2/gen_metadata_qwen3_22302_23364.py >> 6.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="6" nohup python worker2/gen_metadata_qwen3_23364_24426.py >> 7.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="7" nohup python worker2/gen_metadata_qwen3_24426_25488.py >> 8.out 2>&1 &

echo "All 8 jobs started on worker2"

