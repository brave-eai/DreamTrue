#!/bin/bash
set -e -o pipefail

# 用法: ./redis-tmux.sh [PORT] [HOST]
#   左窗格: redis-server
#   右窗格: watch 监控 DBSIZE / KEYS
PORT="${1:-23799}"
HOST="${2:-127.0.0.1}"
SESSION="redis-${PORT}"

# 已存在同名会话则直接 attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
fi

# 左窗格启动 server
tmux new-session -d -s "$SESSION" -n redis \
  "redis-server --bind 0.0.0.0 --protected-mode no --port '$PORT'"

# 右窗格等 server 起来后启动 watch
tmux split-window -h -t "$SESSION" \
  "sleep 1; watch -n 1 -d \"redis-cli -h '$HOST' -p '$PORT' DBSIZE && redis-cli -h '$HOST' -p '$PORT' KEYS '*'\""

tmux select-layout -t "$SESSION" even-horizontal
exec tmux attach -t "$SESSION" -r
