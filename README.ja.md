<div align="center">

<img src="assets/logo.png" alt="Mistral GLM Bridge ロゴ" width="120"/>

# 🌉 Mistral GLM Bridge

**OpenAI 互換 → Mistral `/v1/conversations` プロキシ (9router 用)**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)

</div>

---

## 🌍 README 言語

| 🌐 | 言語 | |
|----|------|---|
| 🇬🇧 | **English** | [読む](README.md) |
| 🇮🇩 | **Bahasa Indonesia** | [Baca](README.id.md) |
| 🇨🇳 | **中文** | [阅读](README.zh.md) |
| 🇯🇵 | **日本語** | [読む](README.ja.md) |
| 🇰🇷 | **한국어** | [읽기](README.ko.md) |

---

## 📌 概要

Mistral には**2つの異なるエンドポイント**があります：

- `/v1/chat/completions`（OpenAI 互換）— サードパーティモデルは**レート制限 (429) になりやすい**
- `/v1/conversations`（Mistral ネイティブ）— **レート制限なし**、スムーズ

**9router** は OpenAI 互換フォーマットを必要とします。このブリッジは：

1. 9router から OpenAI フォーマットのリクエストを受け取る
2. `/v1/conversations` に変換
3. レスポンスを OpenAI フォーマットに変換し直す
4. ローカルポート（デフォルト `8090`）で返す

結果：**GLM-5.2 を Mistral 経由で 429 なし**、通常の 9router プロバイダーとして接続。

## 🚀 クイックスタート

```bash
git clone https://github.com/0xgetz/mistral-bridge.git
cd mistral-bridge
pip install -r requirements.txt

# API キーを設定（デフォルトなし — キーをコミットしないでください）
echo "MISTRAL_KEY=sk-..." > .env

# ブリッジ起動 + 9router ノード確認
./mistral-bridge.sh start
```

## 🛠 コマンド

```bash
./mistral-bridge.sh start     # ブリッジ起動 + 9router ノード確認
./mistral-bridge.sh stop      # ブリッジ停止
./mistral-bridge.sh status    # ブリッジ + ノード状態確認
./mistral-bridge.sh watch     # ウォッチドッグ：クラッシュ時に自動再起動
```

## 📁 構造

```
mistral-bridge/
├── server.py              # FastAPI ブリッジサーバー
├── mistral-bridge.sh      # 制御スクリプト (start/stop/status/watch)
├── assets/                # ロゴ (SVG + PNG)
├── requirements.txt       # 依存関係
├── LICENSE                # MIT
└── README.md              # このファイル (+ .id/.zh/.ja/.ko 翻訳)
```

## 🔧 テスト

```bash
# ブリッジ直接テスト
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'

# 9router 経由
curl http://127.0.0.1:20128/v1/chat/completions \
  -H "Authorization: Bearer <9router-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-compatible-chat-xxxx/glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'
```

## 📦 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `MISTRAL_KEY` | *(必須)* | Mistral API キー |
| `BRIDGE_MODEL` | `glm-5-2` | 転送するモデル |
| `BRIDGE_PORT` | `8090` | ローカル待受ポート |
| `BRIDGE_HOST` | `0.0.0.0` | 待受アドレス |

## 🔄 再起動時の自動起動

```bash
crontab -e
# 追加：
@reboot /root/mistral-bridge/mistral-bridge.sh start
```

## 🤝 貢献

PR 大歓迎！次のルールに従ってください：
- シンプルに保つ (KISS)
- コードに秘密情報を入れない — 常に環境変数を使用
- 提出前にテスト

## 📄 ライセンス

[MIT](LICENSE) © 2026 [0xgetz](https://github.com/0xgetz)