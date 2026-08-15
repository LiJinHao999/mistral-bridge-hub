<div align="center">

<img src="assets/logo.png" alt="Mistral GLM Bridge 标志" width="120"/>

# 🌉 Mistral GLM Bridge

**OpenAI 兼容 → Mistral `/v1/conversations` 代理 (for 9router)**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)

</div>

---

## 🌍 README 语言

| 🌐 | 语言 | |
|----|------|---|
| 🇬🇧 | **English** | [阅读](README.md) |
| 🇮🇩 | **Bahasa Indonesia** | [Baca](README.id.md) |
| 🇨🇳 | **中文** | [阅读](README.zh.md) |
| 🇯🇵 | **日本語** | [読む](README.ja.md) |
| 🇰🇷 | **한국어** | [읽기](README.ko.md) |

---

## 📌 简介

Mistral 有**两个不同的端点**：

- `/v1/chat/completions`（OpenAI 兼容）— 第三方模型**经常被限流 (429)**
- `/v1/conversations`（Mistral 原生）— **无限流**，流畅

**9router** 需要 OpenAI 兼容格式。此桥接器：

1. 接收来自 9router 的 OpenAI 格式请求
2. 翻译为 `/v1/conversations`
3. 将响应翻译回 OpenAI 格式
4. 通过本地端口返回（默认 `8090`）

结果：**GLM-5.2 通过 Mistral 无 429**，作为常规 9router 提供商接入。

## 🚀 快速开始

```bash
git clone https://github.com/0xgetz/mistral-bridge.git
cd mistral-bridge
pip install -r requirements.txt

# 设置 API 密钥（无默认值 — 不要提交密钥）
echo "MISTRAL_KEY=sk-..." > .env

# 启动桥接 + 确保 9router 节点
./mistral-bridge.sh start
```

## 🛠 命令

```bash
./mistral-bridge.sh start     # 启动桥接 + 确保 9router 节点
./mistral-bridge.sh stop      # 停止桥接
./mistral-bridge.sh status    # 检查桥接 + 节点状态
./mistral-bridge.sh watch     # 看门狗：崩溃时自动重启
```

## 📁 结构

```
mistral-bridge/
├── server.py              # FastAPI 桥接服务器
├── mistral-bridge.sh      # 控制脚本 (start/stop/status/watch)
├── assets/                # 标志 (SVG + PNG)
├── requirements.txt       # 依赖
├── LICENSE                # MIT
└── README.md              # 本文件 (+ .id/.zh/.ja/.ko 翻译)
```

## 🔧 测试

```bash
# 直接测试桥接
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'

# 通过 9router
curl http://127.0.0.1:20128/v1/chat/completions \
  -H "Authorization: Bearer <9router-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-compatible-chat-xxxx/glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'
```

## 📦 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MISTRAL_KEY` | *(必需)* | Mistral API 密钥 |
| `BRIDGE_MODEL` | `glm-5-2` | 转发的模型 |
| `BRIDGE_PORT` | `8090` | 本地监听端口 |
| `BRIDGE_HOST` | `0.0.0.0` | 监听地址 |

## 🔄 重启自动启动

```bash
crontab -e
# 添加:
@reboot /root/mistral-bridge/mistral-bridge.sh start
```

## 🤝 贡献

欢迎 PR！请遵循：
- 保持简单 (KISS)
- 代码中不留密钥 — 始终使用环境变量
- 提交前先测试

## 📄 许可证

[MIT](LICENSE) © 2026 [0xgetz](https://github.com/0xgetz)