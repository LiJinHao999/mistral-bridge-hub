<div align="center">

<img src="assets/logo.png" alt="Mistral GLM Bridge 로고" width="120"/>

# 🌉 Mistral GLM Bridge

**OpenAI 호환 → Mistral `/v1/conversations` 프록시 (9router용)**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/built%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/)

</div>

---

## 🌍 README 언어

| 🌐 | 언어 | |
|----|------|---|
| 🇬🇧 | **English** | [읽기](README.md) |
| 🇮🇩 | **Bahasa Indonesia** | [Baca](README.id.md) |
| 🇨🇳 | **中文** | [阅读](README.zh.md) |
| 🇯🇵 | **日本語** | [読む](README.ja.md) |
| 🇰🇷 | **한국어** | [읽기](README.ko.md) |

---

## 📌 개요

Mistral에는 **두 가지 다른 엔드포인트**가 있습니다:

- `/v1/chat/completions` (OpenAI 호환) — 타사 모델은 **레이트 제한 (429) 발생 빈도 높음**
- `/v1/conversations` (Mistral 네이티브) — **레이트 제한 없음**, 원활함

**9router**는 OpenAI 호환 형식이 필요합니다. 이 브리지는:

1. 9router에서 OpenAI 형식 요청을 받음
2. `/v1/conversations`로 변환
3. 응답을 OpenAI 형식으로 다시 변환
4. 로컬 포트(기본 `8090`)로 반환

결과: **GLM-5.2를 Mistral 경유로 429 없이**, 일반 9router 공급자로 연결.

## 🚀 빠른 시작

```bash
git clone https://github.com/0xgetz/mistral-bridge.git
cd mistral-bridge
pip install -r requirements.txt

# API 키 설정 (기본값 없음 — 키를 커밋하지 마세요)
echo "MISTRAL_KEY=sk-..." > .env

# 브리지 시작 + 9router 노드 확인
./mistral-bridge.sh start
```

## 🛠 명령어

```bash
./mistral-bridge.sh start     # 브리지 시작 + 9router 노드 확인
./mistral-bridge.sh stop      # 브리지 중지
./mistral-bridge.sh status    # 브리지 + 노드 상태 확인
./mistral-bridge.sh watch     # 워치독: 충돌 시 자동 재시작
```

## 📁 구조

```
mistral-bridge/
├── server.py              # FastAPI 브리지 서버
├── mistral-bridge.sh      # 제어 스크립트 (start/stop/status/watch)
├── assets/                # 로고 (SVG + PNG)
├── requirements.txt       # 의존성
├── LICENSE                # MIT
└── README.md              # 이 파일 (+ .id/.zh/.ja/.ko 번역)
```

## 🔧 테스트

```bash
# 브리지 직접 테스트
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'

# 9router 경유
curl http://127.0.0.1:20128/v1/chat/completions \
  -H "Authorization: Bearer <9router-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"openai-compatible-chat-xxxx/glm-5-2","messages":[{"role":"user","content":"halo"}],"max_tokens":50}'
```

## 📦 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MISTRAL_KEY` | *(필수)* | Mistral API 키 |
| `BRIDGE_MODEL` | `glm-5-2` | 전달할 모델 |
| `BRIDGE_PORT` | `8090` | 로컬 리슨 포트 |
| `BRIDGE_HOST` | `0.0.0.0` | 리슨 주소 |

## 🔄 재부팅 시 자동 시작

```bash
crontab -e
# 추가:
@reboot /root/mistral-bridge/mistral-bridge.sh start
```

## 🤝 기여

PR 환영합니다! 다음을 지켜주세요:
- 단순하게 유지 (KISS)
- 코드에 비밀 정보 없이 — 항상 환경 변수 사용
- 제출 전 테스트

## 📄 라이선스

[MIT](LICENSE) © 2026 [0xgetz](https://github.com/0xgetz)