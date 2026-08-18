#!/bin/bash
# Control script for the local Mistral GLM Bridge gateway.
#
#   ./mistral-bridge.sh start      start in the background
#   ./mistral-bridge.sh stop       stop
#   ./mistral-bridge.sh restart    stop then start
#   ./mistral-bridge.sh status     show pid / health
#   ./mistral-bridge.sh enable     install a systemd user unit (starts on login)
#   ./mistral-bridge.sh disable    remove that unit
#
# Paths are relative to this script. `.env` in the same directory is loaded
# if present (MISTRAL_KEY, BRIDGE_PORT, BRIDGE_HOST, BRIDGE_MODEL).

set -u

BRIDGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${BRIDGE_DIR}/.env"
LOG_FILE="${BRIDGE_DIR}/bridge.log"
PID_FILE="${BRIDGE_DIR}/.bridge.pid"
UNIT_NAME="mistral-bridge.service"
UNIT_FILE="${HOME}/.config/systemd/user/${UNIT_NAME}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

export MISTRAL_KEY="${MISTRAL_KEY:-}"
BRIDGE_PORT="${BRIDGE_PORT:-8577}"
BRIDGE_HOST="${BRIDGE_HOST:-0.0.0.0}"

if [ -x "${BRIDGE_DIR}/.venv/bin/python" ]; then
  PYTHON="${BRIDGE_DIR}/.venv/bin/python"
else
  PYTHON="$(command -v python3 || true)"
fi

health_url() {
  echo "http://127.0.0.1:${BRIDGE_PORT}/health"
}

bridge_up() {
  curl -sf -m 5 "$(health_url)" >/dev/null 2>&1
}

running_pid() {
  local pid=""
  if [ -f "$PID_FILE" ]; then
    pid="$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  pid="$(pgrep -u "$(id -u)" -f "${BRIDGE_DIR}/server.py" | head -1 || true)"
  if [ -n "$pid" ]; then
    echo "$pid"
    return 0
  fi
  pid="$(ss -lntpH "sport = :${BRIDGE_PORT}" 2>/dev/null \
         | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [ -n "$pid" ] && echo "$pid"
}

start_bridge() {
  if bridge_up; then
    echo "[bridge] already running on :${BRIDGE_PORT} (pid $(running_pid))"
    return 0
  fi
  if [ -z "$PYTHON" ]; then
    echo "[bridge] python3 not found. Create a venv first:"
    echo "  python3 -m venv ${BRIDGE_DIR}/.venv"
    echo "  ${BRIDGE_DIR}/.venv/bin/pip install -r ${BRIDGE_DIR}/requirements.txt"
    return 1
  fi

  echo "[bridge] starting with ${PYTHON} on ${BRIDGE_HOST}:${BRIDGE_PORT}"
  # Python RotatingFileHandler owns bridge.log. Do not also nohup-append to it:
  # a rotate would rename the file while this fd kept writing the old inode.
  setsid nohup "$PYTHON" -u "${BRIDGE_DIR}/server.py" >> /dev/null 2>&1 < /dev/null &
  echo $! > "$PID_FILE"

  local attempt
  for attempt in $(seq 1 20); do
    bridge_up && break
    sleep 0.5
  done
  if bridge_up; then
    echo "[bridge] up on :${BRIDGE_PORT} (pid $(cat "$PID_FILE"))"
  else
    echo "[bridge] failed to start — see ${LOG_FILE}"
    tail -20 "$LOG_FILE" 2>/dev/null || true
    return 1
  fi
}

stop_bridge() {
  local pid
  pid="$(running_pid)"
  if [ -z "$pid" ]; then
    echo "[bridge] not running"
    rm -f "$PID_FILE"
    return 0
  fi
  kill "$pid" 2>/dev/null || true
  local attempt
  for attempt in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.3
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[bridge] still running, sending SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[bridge] stopped (pid $pid)"
}

show_status() {
  if bridge_up; then
    echo "[bridge] up :${BRIDGE_PORT} (pid $(running_pid))"
    curl -s -m 5 "$(health_url)"
    echo
  else
    echo "[bridge] down :${BRIDGE_PORT}"
    return 1
  fi
}

enable_boot() {
  if ! command -v systemctl >/dev/null 2>&1; then
    echo "[bridge] systemd not found. To start on reboot, add a crontab line:"
    echo "  @reboot ${BRIDGE_DIR}/mistral-bridge.sh start"
    return 1
  fi
  mkdir -p "$(dirname "$UNIT_FILE")"
  cat > "$UNIT_FILE" <<EOF
[Unit]
Description=Mistral GLM Bridge
After=network.target

[Service]
Type=simple
WorkingDirectory=${BRIDGE_DIR}
EnvironmentFile=-${ENV_FILE}
ExecStart=${PYTHON:-/usr/bin/python3} -u ${BRIDGE_DIR}/server.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now "$UNIT_NAME"
  echo "[bridge] systemd user unit enabled: ${UNIT_FILE}"
  echo "[bridge] it will start on login. For a headless box also run:"
  echo "  sudo loginctl enable-linger $(id -un)"
}

disable_boot() {
  if [ ! -f "$UNIT_FILE" ]; then
    echo "[bridge] no systemd user unit installed"
    return 0
  fi
  systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
  rm -f "$UNIT_FILE"
  systemctl --user daemon-reload 2>/dev/null || true
  echo "[bridge] systemd user unit removed"
}

case "${1:-start}" in
  start)
    start_bridge
    ;;
  stop)
    stop_bridge
    ;;
  restart)
    stop_bridge
    start_bridge
    ;;
  status)
    show_status
    ;;
  enable)
    enable_boot
    ;;
  disable)
    disable_boot
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|enable|disable}"
    exit 1
    ;;
esac
