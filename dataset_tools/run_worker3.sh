#!/bin/bash
# 服务器4：运行第25-32个脚本（25488-33990），使用显卡0-7


ASCEND_RT_VISIBLE_DEVICES="0" nohup python worker3/gen_metadata_qwen3_25488_26550.py >> 1.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="1" nohup python worker3/gen_metadata_qwen3_26550_27612.py >> 2.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="2" nohup python worker3/gen_metadata_qwen3_27612_28674.py >> 3.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="3" nohup python worker3/gen_metadata_qwen3_28674_29736.py >> 4.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="4" nohup python worker3/gen_metadata_qwen3_29736_30798.py >> 5.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="5" nohup python worker3/gen_metadata_qwen3_30798_31860.py >> 6.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="6" nohup python worker3/gen_metadata_qwen3_31860_32922.py >> 7.out 2>&1 &
ASCEND_RT_VISIBLE_DEVICES="7" nohup python worker3/gen_metadata_qwen3_32922_33990.py >> 8.out 2>&1 &

echo "All 8 jobs started on worker3"

