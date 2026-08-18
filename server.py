#!/usr/bin/env python3
"""
Mistral GLM Bridge — OpenAI-compatible -> Mistral /v1/conversations

Translates:
  POST /v1/chat/completions  (stateless: full messages, always create)
  POST /v1/responses         (stateful: input + previous_response_id → append)

Env vars (all optional except a key from the client or MISTRAL_KEY):
    MISTRAL_KEY     Mistral API key (no default)
    BRIDGE_MODEL    model id (default: glm-5-2)
    BRIDGE_PORT     listen port (default: 8090)
    BRIDGE_HOST     listen host (default: 0.0.0.0)
"""

import uvicorn

from bridge.app import app
from bridge.config import HOST, MODEL, PORT, log

if __name__ == "__main__":
    log.info("Mistral bridge on %s:%d, model=%s", HOST, PORT, MODEL)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
