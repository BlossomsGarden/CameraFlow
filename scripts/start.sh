conda activate unifiedrewardqwen3
cd /home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO/flow_grpo/server_patch
bash start_qwen3vl_servers.sh

conda activate da3
bash start_da3_servers.sh

conda activate flowgrpo
cd /home/ma-user/modelarts/user-job-dir/wlh/Code/FlowGRPO/
bash scripts/multi_node/recam-24/main.sh

