#!/bin/bash
# mistral-bridge.sh — auto-start + watchdog + auto-colok node 9router
# Usage:
#   ./mistral-bridge.sh start   # start bridge (screen) + pastikan node 9router ada
#   ./mistral-bridge.sh stop    # stop bridge
#   ./mistral-bridge.sh status  # cek status
#   ./mistral-bridge.sh watch   # watchdog loop (jalan terus, restart kalau mati)

BRIDGE_DIR="/root/mistral-bridge"
BRIDGE_PORT="8090"
BRIDGE_MODEL="glm-5-2"
NODE_ID="openai-compatible-chat-a43c985c"   # node yang udah terdaftar
DB="/root/.9router/db/data.sqlite"
ENV_FILE="${BRIDGE_DIR}/.env"                # isi: MISTRAL_KEY=sk-...

bridge_up() {
  curl -s -m 5 "http://127.0.0.1:${BRIDGE_PORT}/v1/models" >/dev/null 2>&1
}

# Load key dari .env (kalau ada)
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
export MISTRAL_KEY="${MISTRAL_KEY:-}"

start_bridge() {
  if bridge_up; then
    echo "[bridge] sudah jalan di :${BRIDGE_PORT}"
  elif [ -z "$MISTRAL_KEY" ]; then
    echo "[bridge] MISTRAL_KEY kosong — isi ${ENV_FILE} dulu:"
    echo "  echo 'MISTRAL_KEY=sk-xxx' >> ${ENV_FILE}"
  else
    echo "[bridge] start..."
    screen -dmS mistral-bridge bash -c "cd ${BRIDGE_DIR} && exec python3 -u server.py > bridge.log 2>&1"
    sleep 5
    if bridge_up; then
      echo "[bridge] OK di :${BRIDGE_PORT}"
    else
      echo "[bridge] GAGAL start — cek ${BRIDGE_DIR}/bridge.log"
      tail -5 "${BRIDGE_DIR}/bridge.log" 2>/dev/null
    fi
  fi
}

ensure_node() {
  # Cek node di DB 9router; kalau hilang, bikin ulang (idempotent)
  if python3 -c "
import sqlite3, json
c = sqlite3.connect('$DB')
exists = c.execute(\"SELECT 1 FROM providerNodes WHERE id='$NODE_ID'\").fetchone()
print('yes' if exists else 'no')
c.close()
" 2>/dev/null | grep -q yes; then
    echo "[node] $NODE_ID sudah ada ✅"
  else
    echo "[node] bikin node + connection + kv..."
    python3 - <<PYEOF
import sqlite3, json, uuid, datetime
c = sqlite3.connect('$DB')
cur = c.cursor()
now = datetime.datetime.utcnow().isoformat()+'Z'
node_id = '$NODE_ID'
cur.execute("INSERT INTO providerNodes (id, type, name, data, createdAt, updatedAt) VALUES (?,?,?,?,?,?)",
    (node_id, 'openai-compatible', 'Mistral GLM bridge',
     json.dumps({'baseUrl': 'http://127.0.0.1:$BRIDGE_PORT/v1', 'description': 'GLM-5.2 via Mistral /v1/conversations bridge'}),
     now, now))
conn_id = str(uuid.uuid4())
cur.execute("INSERT INTO providerConnections (id, provider, authType, name, email, priority, isActive, data, createdAt, updatedAt) VALUES (?,?,?,?,?,?,?,?,?,?)",
    (conn_id, node_id, 'apiKey', 'mistral-glm', 'api@mistral.ai', 0, 1,
     json.dumps({'apiKey': 'not-needed',
                 'providerSpecificData': {'prefix': 'mistralglm', 'apiType': 'chat', 'baseUrl': 'http://127.0.0.1:$BRIDGE_PORT/v1', 'nodeName': 'Mistral GLM bridge', 'models': ['$BRIDGE_MODEL']},
                 'testStatus': 'active'}),
     now, now))
cur.execute("INSERT OR REPLACE INTO kv (scope, key, value) VALUES (?,?,?)",
    ('customModels', f"{node_id}|$BRIDGE_MODEL|llm",
     json.dumps({"providerAlias": node_id, "id": "$BRIDGE_MODEL", "type": "llm", "name": "$BRIDGE_MODEL"})))
c.commit(); c.close()
PYEOF
    echo "[node] selesai ✅"
  fi
}

case "${1:-start}" in
  start)
    start_bridge
    ensure_node
    echo "── cek via 9router ──"
    curl -s -m 60 "http://127.0.0.1:20128/v1/chat/completions" \
      -H "Authorization: Bearer sk-4c4aba6d1a5a5e42-ug83sc-09c309e7" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${NODE_ID}/${BRIDGE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"halo\"}],\"max_tokens\":40}" | head -c 250
    echo
    ;;
  stop)
    screen -S mistral-bridge -X quit 2>/dev/null
    echo "[bridge] stopped"
    ;;
  status)
    bridge_up && echo "[bridge] UP :${BRIDGE_PORT}" || echo "[bridge] DOWN"
    python3 -c "
import sqlite3
c = sqlite3.connect('$DB')
print('[node]', 'ADA' if c.execute(\"SELECT 1 FROM providerNodes WHERE id='$NODE_ID'\").fetchone() else 'HILANG')
c.close()" 2>/dev/null
    ;;
  watch)
    echo "[watch] loop — Ctrl+C stop. Restart otomatis kalau bridge mati."
    while true; do
      if ! bridge_up; then
        echo "[$(date +%H:%M:%S)] bridge mati → restart"
        start_bridge
      fi
      sleep 15
    done
    ;;
  *)
    echo "Usage: $0 {start|stop|status|watch}"
    ;;
esac