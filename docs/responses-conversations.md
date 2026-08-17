# Responses → Conversations：过程记录

日期：2026-08-17  
范围：只记协议对照和改法，不改代码。  
结论：**Mistral Conversations 是 Responses 一类的服务端线程 API，不是 Chat Completions。桥接应只做请求体/响应体字段翻译，不要自建会话状态机。**

---

## 1. 为什么前一版会越改越复杂

`/v1/responses` 客户端（Cursor / Claude Code）经常：

- 不带 `previous_response_id`
- 把整窗历史塞进 `input`（首条常是 `<user_info>`）
- 看到 `status: incomplete` 就重试同一问

上一版用 fingerprint / replay / inflight / pending-tool 表去「猜」同一聊天。这套东西：

- 不是 Responses 原生语义
- 会把客户端本地 UUID 误当成 Mistral `conversation_id` 去 append（08:28 日志：`257023a2-…` → 上游 404 code 3000）
- stream 非 200 不清理 inflight，请求卡死约 2 分钟

用户纠正：Conversations 就是 Responses 衍生格式，**兼容好 Responses、改好请求体即可**。不要再堆会话缓存。

---

## 2. 两边各自是什么

| | OpenAI Responses | Mistral Conversations |
|---|---|---|
| 端点 | `POST /v1/responses` | 创建 `POST /v1/conversations`；续写 `POST /v1/conversations/{conversation_id}` |
| 身份 | `response.id`（本轮） | `conversation_id`（整条线程） |
| 续写指针 | `previous_response_id` 或 `conversation` / `conversation.id` | 路径上的 `{conversation_id}` |
| 本轮内容 | `input`（string 或 item 数组） | `inputs`（string 或 entry 数组） |
| 是否落盘 | `store`（默认 true，约 30 天） | `store`（默认 true） |
| 系统提示 | `instructions` | `instructions`（**仅创建**；append 禁止） |
| 工具定义 | `tools` | `tools`（**仅创建**；append 禁止） |
| 采样 | 顶层 `temperature` / `max_output_tokens` / `tool_choice` / `reasoning` | 一律进 `completion_args` |
| 对象类型 | `object: "response"` | `object: "conversation.response"` |

官方文档没有写「Conversations 派生自 OpenAI Responses」。行为上是同一类：服务端存线程，客户端只交本轮 `input` + 上一轮 id。

Mistral **没有** `/v1/responses`。桥的 `/v1/responses` 是翻译层。

来源：

- https://docs.mistral.ai/api/endpoint/beta/conversations
- https://docs.mistral.ai/studio-api/agents/agents-api
- https://docs.mistral.ai/studio-api/agents/agent-tools/function-calling

---

## 3. 请求体对照（核心）

### 3.1 顶层字段

| Responses | Conversations 创建 | Conversations append |
|---|---|---|
| `model` | `model` | **禁止**（`strip`） |
| `input` | `inputs` | `inputs`（只含本轮新条目） |
| `instructions` | `instructions` | **禁止** |
| `tools` | `tools` | **禁止** |
| `store` | `store`（原样转发，不要写死 true） | `store` |
| `stream` | `stream` | `stream` |
| `previous_response_id` / `conversation` / `conversation.id` | — | 变成 URL 路径 id |
| `temperature` `top_p` `max_output_tokens` `max_tokens` `tool_choice` `reasoning` / `reasoning_effort` | `completion_args.*` | `completion_args.*` |
| `metadata` | `metadata` | 无（忽略） |

append 官方允许的字段只有：`inputs`、`completion_args`、`store`、`stream`、`handoff_execution`、`tool_confirmations`。带上 `model` / `instructions` / `tools` 会被拒。

### 3.2 `input` item → `inputs` entry

| Responses item | Conversations entry |
|---|---|
| `"hello"` 或 `{role:"user", content:"hello"}` | `{role:"user", content:"hello"}` |
| `{type:"message", role:"assistant", content:[{type:"output_text", text:"…"}]}` | `{role:"assistant", content:"…"}` |
| `{type:"function_call", call_id, name, arguments}` | `{type:"function.call", tool_call_id, name, arguments}` |
| `{type:"function_call_output", call_id, output}` | `{type:"function.result", tool_call_id, result}` |
| `{role:"tool", tool_call_id, content}` | 同上 `function.result` |
| `{role:"system"\|"developer", content}` | 并入 `instructions`，不进 `inputs` |
| `{type:"reasoning", …}` | 丢掉（Conversations 用 thinking 块，不吃 reasoning item） |

`function.result` 官方示例：

```json
{
  "object": "entry",
  "type": "function.result",
  "tool_call_id": "<id>",
  "result": "<string>"
}
```

`object: "entry"` 可省。`result` 必须是字符串。

### 3.3 路由

```
previous_id 为空     → POST /v1/conversations          （create）
previous_id 非空     → POST /v1/conversations/{id}     （append）
append 返回 404/3000 → 同一 payload 改打 create        （id 不是本桥发出的 conv）
```

`previous_id` 只认客户端显式字段：

1. `previous_response_id`
2. `conversation_id`
3. `conversation`（string）或 `conversation.id` / `conversation.conversation_id`
4. 头 `X-Conversation-Id` / `X-Previous-Response-Id`

**不要**用 fingerprint、首条 user、pending tool 表去猜。

客户端本地 UUID（如 `257023a2-f57c-4e1b-9122-b7a2d67e4fea`）不是 Mistral 会话。先 append，404/3000 再 create。不要预先按 id 形态过滤——Mistral 自己的 id 也有 UUID 形态。

### 3.4 `store`

原样转发。客户端 `store: false` 则本轮不落盘，后续 `previous_response_id` 无法续写——这是原生行为，不要改。

---

## 4. 响应体对照

### 4.1 非流式

Mistral：

```json
{
  "object": "conversation.response",
  "conversation_id": "conv_… 或 UUID",
  "outputs": [ /* MessageOutputEntry | FunctionCallEntry | … */ ],
  "usage": { "prompt_tokens", "completion_tokens", "total_tokens" }
}
```

译成 Responses：

```json
{
  "id": "<conversation_id>",
  "object": "response",
  "created_at": <unix>,
  "model": "<请求的 model>",
  "status": "completed",
  "output": [ /* 见下表 */ ],
  "usage": {
    "input_tokens": prompt_tokens,
    "output_tokens": completion_tokens,
    "total_tokens": total_tokens
  }
}
```

**`response.id` 必须等于 `conversation_id`。** 客户端下一轮会把它当作 `previous_response_id` 原样送回。

### 4.2 `outputs[]` → `output[]`

| Mistral output | Responses output item |
|---|---|
| `{type:"message.output", role:"assistant", content:[text\|thinking]}` | `{type:"message", id, role:"assistant", status:"completed", content:[{type:"output_text", text}]}`；thinking 另出 `{type:"reasoning", summary:[{type:"summary_text", text}]}` |
| `{type:"function.call", tool_call_id, name, arguments}` | `{type:"function_call", id, call_id: tool_call_id, name, arguments}` |

`function.call` 也是**完整的一轮**。`status` 必须是 `completed`。标 `incomplete` 会让客户端重试同一问。

### 4.3 流式事件（最小集）

上游 SSE：`conversation.response.started`（带 `conversation_id`）→ `message.output.delta` / `function.call.delta` → `conversation.response.done`。

下游至少发：

1. `response.created`（`response.id = conversation_id`，越早越好）
2. `response.output_item.added` / `response.output_text.delta` 或 `response.function_call_arguments.delta`
3. `response.output_item.done`
4. `response.completed`（`status: "completed"`，`id` 仍是同一个 `conversation_id`）

出错：`response.failed`，立刻结束，不挂 inflight。

---

## 5. 目标数据流（改完后应只剩这个）

```
客户端 POST /v1/responses
        │  body: { model, input, previous_response_id?, store?, tools?, instructions?, stream? }
        ▼
翻译请求体
        │  input → inputs（条目改名）
        │  采样字段 → completion_args
        │  有 previous_id → URL = /v1/conversations/{id}，并从 body 去掉 model/instructions/tools
        │  无 previous_id → URL = /v1/conversations
        ▼
Mistral
        │  404/3000 且刚才是 append → 改 POST /v1/conversations 重试一次
        ▼
翻译响应体
        │  conversation_id → response.id
        │  outputs → output
        │  status = completed
        ▼
客户端
```

Chat Completions 路径（`/v1/chat/completions`）是另一套：客户端每次带全量 `messages`。那边应继续 `store: false` + 每次 create，或把全量 `messages` 译成一次 create 的 `inputs`。不要和 Responses 共用 fingerprint 表。

---

## 6. 现有 `server.py` 里该删 / 该留

相对 2026-08-17 工作区未提交稿。

**删掉（不是 Responses 语义）：**

- `SESSIONS` / `CONV_META` / `PENDING_CALLS` / `PENDING_BY_CONV`
- `INFLIGHT` / `INFLIGHT_DONE` / `CREATING`
- `session_fingerprint` / `_is_boilerplate` / `_text_digest`
- `plan_upstream` / `lookup_session` / `remember_conversation` / `forget_conversation`
- `replay_*` / `wait_for_*` / `mark_inflight` / `clear_inflight`
- `apply_conversation_plan` / `resolve_plan`

**留下并收干净：**

- `responses_input_to_entries` / `responses_to_mistral`（补：`store` 原样转发）
- `mistral_to_responses` / `stream_responses`（`id = conversation_id`，`status = completed`）
- `strip_for_append`（append 只留允许字段）
- `extract_previous_id`（只读显式字段）
- 路由：有 id → append；404/3000 → create 一次

**不要再做：**

- 用首条 `<user_info>` 或「第一条真问题」猜同一窗
- 同文重试回放缓存
- 等 inflight / 串行化 create
- 把非本进程发过的 id 直接当死 id 丢掉（先 append，让上游判）

---

## 7. 已知边界（文档写明，实现时不要再发明）

1. **客户端不带 `previous_response_id`、每次把全窗当 `input`。**  
   原生 Responses 也是每次新建。桥不能靠猜修复。先保证返回的 `response.id` 能被下一轮带回来；若客户端仍不带，那是客户端问题，记日志观察，不要回指纹。

2. **append 时 `input` 里混了旧历史。**  
   原生语义：`input` = 本轮新条目。客户端按规范应只发新 turn。桥不要自己裁「最后一条 user」——那是猜。若日志证明 Cursor 在带 `previous_response_id` 时仍塞全窗，另开一条再议，不要和「无 id」混在一起修。

3. **Mistral：append 时若会话在等 `function.result`，再塞 user 会 3001。**  
   规范做法：本轮 `input` 里该有对应的 `function_call_output`。不要为了躲 3001 丢掉会话另建——那会丢掉工具上下文，正是「反复回答同一问」的来源之一。

4. **stream 断开 / 上游非 200。**  
   直接 `response.failed` 结束。不要留任何进程内锁。

---

## 8. 验收（实现后对照，现在不跑）

1. `POST /v1/responses` `{"input":"halo"}` → `id` 为上游 `conversation_id`，`status=completed`。
2. 下一轮带 `previous_response_id=<上一个 id>` + 新 `input` → 日志为 append，上游 URL 含该 id，模型能看见上一轮。
3. `previous_response_id` 为随机 UUID → 上游 404/3000 → 自动 create，返回新 `conversation_id`，不卡死。
4. 模型回 `function_call` → 流式/非流式都是 `completed`，`call_id` 可在下一轮 `function_call_output` 里对上。
5. 进程内无 fingerprint / session 字典；重启后只要客户端带着上次的 `response.id`，append 仍能打到 Mistral。

---

## 9. 日志里已经验证过的失败（用来对照实现）

```
08:28:58  append → 257023a2-…  plan=next_user  sessions=0
08:28:59  upstream 404  Conversation … was not found  (code 3000)
08:28:59  wait for in-flight create  → 45s 超时
08:29:44  wait for in-flight turn    → 90s 超时
08:31:14  plan=wait_inflight
```

正确行为应是：08:28:59 收到 3000 后立刻改 POST `/v1/conversations` create，数秒内返回新 `conversation_id`。
