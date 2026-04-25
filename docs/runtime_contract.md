# Runtime Contract Guard

这份文档记录 runtime 对产品层必须稳定输出的契约。多 provider 适配可以改变“怎么向 provider 发请求、怎么解析 provider 返回”，但不能改变前端、历史记录、上下文工作区看到的产品语义。

## Transcript 产品语义

一次主聊天回合无论有没有工具调用，最终都只能向产品 transcript 新增两条记录：

1. `role: "user"`：用户这轮输入。
2. `role: "assistant"`：助手这轮最终回答，以及这一轮内部发生的工具事件。

也就是说，无工具、单工具、多工具三种情况都不能把 provider 协议层的中间消息拆成额外 transcript 节点。工具调用、工具结果、provider function call/output 只是 assistant 这条产品记录的内部细节。

不能破这个契约的原因很直接：前端的消息列表、上下文地图、撤销/恢复、token 统计和会话标题都把一轮对话理解成 `user + assistant`。如果工具调用被插成额外节点，用户会看到协议噪声，上下文节点编号会漂移，恢复旧快照也会变得不可靠。

## 工具事件折叠位置

工具事件只能折叠在 assistant record 的这些字段里：

- `toolEvents`
- `blocks` 中的 `{ "kind": "tool", "tool_event": ... }`
- `providerItems` 中的 provider 协议细节，例如 `function_call` / `function_call_output`
- 未来如果引入统一格式，放在 `canonicalItems`

产品 transcript 顶层不能出现 `role: "tool"`、`role: "system"`、`role: "developer"` 这类 provider 协议节点。也不能为了保存工具结果而额外生成一条“伪 user”记录。

一个含工具的 assistant record 可以长这样：

```json
{
  "role": "assistant",
  "text": "我查到了结果。",
  "toolEvents": [
    {
      "name": "shell_command",
      "arguments": { "command": "pwd" },
      "output_preview": "C:\\Projects\\hash-code",
      "status": "completed"
    }
  ],
  "blocks": [
    {
      "kind": "tool",
      "tool_event": {
        "name": "shell_command",
        "arguments": { "command": "pwd" },
        "output_preview": "C:\\Projects\\hash-code",
        "status": "completed"
      }
    },
    {
      "kind": "text",
      "text": "我查到了结果。"
    }
  ],
  "providerItems": [
    {
      "type": "function_call",
      "call_id": "stored_1_1",
      "name": "shell_command",
      "arguments": "{\"command\":\"pwd\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "stored_1_1",
      "output": "C:\\Projects\\hash-code"
    },
    {
      "type": "message",
      "role": "assistant",
      "content": "我查到了结果。"
    }
  ]
}
```

## Streaming 前端事件语义

发送给前端的 streaming 事件只能是产品层事件：

- `delta`：追加助手文本片段。
- `reset`：工具轮次导致前一段临时文本要重置。
- `tool_event`：展示一次工具调用的产品化摘要。
- `done`：本轮完成，携带最终 answer、blocks、tool_events、session/sidebar 或上下文工作区结果。
- `error`：本轮失败，携带用户可读错误。

provider 原始事件不能穿透到前端，例如 OpenAI Responses 的 `response.output_text.delta`、`response.output_item.done`，或者其他 provider 的原始 chunk / candidate / content block 事件。适配层必须把它们翻译成上面的五类产品事件。

这个边界同样不能破：前端只应该依赖稳定产品事件。如果直接暴露 provider 原始事件，切换 provider 就会变成一次前端协议迁移，历史流式 UI、工具卡片和错误处理也会被迫认识每个 provider 的细节。

## 测试护栏

契约测试放在 `tests/agent_runtime/`，fixture 放在 `tests/fixtures/`。这些测试不调用真实 API，也不需要真实 API key。后续新增 provider 时，应该优先新增 provider 的 transcript/stream fixture，再让它通过同一批契约校验。
