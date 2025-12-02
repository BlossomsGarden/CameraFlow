#!/bin/bash
# Start Qwen3VL API servers on each RANK/GPU
# Usage: bash start_qwen3vl_servers.sh [base_port] [num_ranks]
# Example: bash start_qwen3vl_servers.sh 34569 8

BASE_PORT=${1:-34575}
NUM_RANKS=${2:-8}

echo "Starting Qwen3VL API servers on ports $BASE_PORT to $((BASE_PORT + NUM_RANKS - 1))"

for rank in $(seq 0 $((NUM_RANKS - 1))); do
    port=$((BASE_PORT + rank))
    echo "Starting server for RANK $rank on port $port"
    
    # Set RANK environment variable and start server in background
    RANK=$rank LOCAL_RANK=$rank ASCEND_RT_VISIBLE_DEVICES=$rank \
        python api_server_qwen3vl.py --base-port $BASE_PORT > qwen3vl_server_rank${rank}.log 2>&1 &
    
    echo "  Server for RANK $rank started (PID: $!)"
    sleep 1  # Small delay to avoid port conflicts
done

echo "All servers started. Check logs: qwen3vl_server_rank*.log"
echo "To stop all servers: pkill -f api_server_qwen3vl.py"

