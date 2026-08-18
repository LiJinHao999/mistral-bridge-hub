<div align="center">

<img src="assets/logo.png" alt="Mistral GLM Bridge 标志" width="120"/>

# Mistral GLM Bridge

**OpenAI 兼容 API → Mistral `/v1/conversations`**

[English](README.md) · [中文](README.zh.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

</div>

基于 [0xgetz/mistral-bridge](https://github.com/0xgetz/mistral-bridge) 的 fork。这是一个本地 OpenAI 兼容代理，让 Cursor、Claude Code 等客户端可以直接打 Mistral 上的模型（默认 `glm-5-2`），而不走 Mistral 经常 429 的 Chat Completions。

## 为什么需要桥

Mistral 有两套接口：

| 端点 | 格式 | 说明 |
|---|---|---|
| `/v1/chat/completions` | OpenAI Chat Completions | 第三方模型经常 **429** |
| `/v1/conversations` | Mistral 原生线程 | create + append，不受 Chat Completions 配额限制 |

本桥接收 OpenAI 形态的请求，转成 Conversations 再把流/JSON 翻译回去。

## 接口

| 方法 | 路径 | 行为 |
|---|---|---|
| `POST` | `/v1/chat/completions` | 无状态。始终用完整 `messages` **新建**会话。 |
| `POST` | `/v1/responses` | 有状态。能匹配到上一轮就 **append**，否则 create。`response.id` 即 Mistral `conversation_id`。 |
| `POST` | `/v1/messages` | Anthropic Messages API。无状态，同 Chat Completions。支持 `system`、`thinking`、`tool_use`/`tool_result` content blocks 和 `x-api-key` 鉴权。 |
| `GET` | `/v1/models` | 透传到 Mistral；失败则回退到本地配置的模型。 |
| `GET` | `/v1/models/{id}` | 同上，单条模型卡片。 |
| `GET` | `/health` | `{status, model, port}` |

鉴权：客户端 `Authorization: Bearer …`（OpenAI）或 `x-api-key: …`（Anthropic）优先，否则用 `MISTRAL_KEY`。

Chat Completions、Responses 和 Messages 都支持 `"stream": true`。

## 快速开始

```bash
git clone https://github.com/LiJinHao999/mistral-bridge-hub.git
cd mistral-bridge-hub
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

echo "MISTRAL_KEY=sk-..." > .env
set -a; . ./.env; set +a

./mistral-bridge.sh start
```

默认监听 `0.0.0.0:8577`。

```bash
curl http://127.0.0.1:8577/v1/chat/completions \
  -H "Authorization: Bearer $MISTRAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'

curl http://127.0.0.1:8577/v1/responses \
  -H "Authorization: Bearer $MISTRAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","input":"hello","max_output_tokens":50}'

curl http://127.0.0.1:8577/v1/messages \
  -H "x-api-key: $MISTRAL_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'
```

任意 OpenAI 或 Anthropic 兼容客户端把 base URL 指到 `http://127.0.0.1:8577/v1`，用同一把 key 即可。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MISTRAL_KEY` | *(无)* | 客户端没带 `Authorization` 时的兜底 key |
| `BRIDGE_MODEL` | `glm-5-2` | 客户端没指定模型时的默认值 |
| `BRIDGE_PORT` | `8577` | 监听端口 |
| `BRIDGE_HOST` | `0.0.0.0` | 监听地址 |

`./mistral-bridge.sh` 会读 `.env`，用来在 Linux 上启停这个网关：

```bash
./mistral-bridge.sh start      # 后台启动
./mistral-bridge.sh stop
./mistral-bridge.sh restart
./mistral-bridge.sh status
./mistral-bridge.sh enable     # 安装 systemd 用户单元，登录后自动拉起
./mistral-bridge.sh disable
```

无图形登录的机器，`enable` 之后再执行 `sudo loginctl enable-linger $USER`，重启才不会丢。没有 systemd 时：

```bash
crontab -e
# @reboot /path/to/mistral-bridge/mistral-bridge.sh start
```

## 目录

```
mistral-bridge/
├── server.py              # uvicorn 入口
├── bridge/
│   ├── config.py          # 环境变量、超时、日志
│   ├── utils.py           # key、文本扁平化、错误响应
│   ├── models.py          # 本地 /v1/models 回退
│   ├── tools.py           # function.call / function.result
│   ├── cache.py           # create vs append 匹配
│   ├── translate.py       # Chat / Responses / Anthropic ↔ Conversations
│   ├── sse.py             # Mistral SSE 解析
│   ├── upstream.py        # create / append / GET
│   ├── streaming.py       # SSE → OpenAI / Anthropic 事件
│   ├── routes.py          # HTTP 路由
│   └── app.py             # FastAPI 应用
├── mistral-bridge.sh      # start/stop/restart/status/enable
├── requirements.txt
└── LICENSE
```

## 许可证

[MIT](LICENSE)。上游版权：0xgetz。
