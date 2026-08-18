"""Runtime configuration for the Mistral GLM Bridge.

Env vars (all optional except a key from the client or MISTRAL_KEY):
    MISTRAL_KEY     Mistral API key (no default)
    BRIDGE_MODEL    model id (default: glm-5-2)
    BRIDGE_PORT     listen port (default: 8090)
    BRIDGE_HOST     listen host (default: 0.0.0.0)
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import aiohttp

MISTRAL_KEY = os.environ.get("MISTRAL_KEY", "")
MODEL = os.environ.get("BRIDGE_MODEL", "glm-5-2")
PORT = int(os.environ.get("BRIDGE_PORT", 8577))
HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
MISTRAL_API = "https://api.mistral.ai/v1"
MISTRAL_BASE = f"{MISTRAL_API}/conversations"
# Wall-clock `total` is a trap for high-reasoning streams: thinking of ~36k
# chars was observed cut at ~273s, which is this timeout minus connect.
# sock_read is the stall detector; each SSE chunk resets it. Connect stays
# short so an unreachable host fails fast.
UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=180)
# Mistral sometimes drops the TCP handshake / first bytes on create.
# Retry locally so AxonHub does not see an empty SSE and replay the turn.
CREATE_CONNECT_RETRIES = 3
CREATE_RETRY_BACKOFF = 0.4
APPEND_CONFLICT_RETRIES = 4
APPEND_CONFLICT_BACKOFF = 0.4
# Mistral prompt-cache blocks are 64 tokens. Conversations usage omits
# cached_tokens, so we floor the local estimate to this size.
CACHE_BLOCK_TOKENS = 64
# On create, replay ThinkChunks from the client window so a dropped
# conversation does not force the model to re-plan from zero. Cap the
# replayed text (most recent first) so a 30k-char ramble cannot double
# an already 40–80k-token create prompt.
THINKING_CREATE_MAX_CHARS = 12000

# nohup used to append stdout forever with no rotation. 83MB in ~90 minutes of
# agent traffic (every tool call/result is INFO). The file handler rotates
# in-process; stderr stays for systemd / an attached terminal.
_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.environ.get("BRIDGE_LOG", os.path.join(_LOG_DIR, "bridge.log"))
_LOG_MAX_BYTES = int(os.environ.get("BRIDGE_LOG_MAX_BYTES", 10 * 1024 * 1024))
_LOG_BACKUPS = int(os.environ.get("BRIDGE_LOG_BACKUPS", 2))
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_handlers = [logging.StreamHandler()]
try:
    _file = RotatingFileHandler(
        LOG_PATH, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS, encoding="utf-8",
    )
    _file.setFormatter(_fmt)
    _handlers.append(_file)
except OSError:
    pass
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=_handlers)
log = logging.getLogger("mistral-bridge")
