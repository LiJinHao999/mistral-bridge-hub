"""Mistral GLM Bridge — OpenAI-compatible API over Mistral Conversations.

Layout:
    config      env, timeouts, logger
    utils       keys, content flattening, HTTP error helper
    models      local /v1/models fallback cards
    tools       function.call / function.result translation
    cache       create-vs-append conversation matching
    translate   Chat Completions / Responses <-> Conversations
    sse         Mistral SSE parser
    upstream    create / append / GET passthrough
    streaming   SSE proxies to OpenAI event shapes
    routes      FastAPI endpoints
    app         FastAPI application
"""

__version__ = "1.0.0"
