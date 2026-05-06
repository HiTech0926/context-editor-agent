# 多接口适配子 Agent 派工方案

## 1. 先说结论

这个任务**可以拆给子 agent 做**，但不能一股脑同时开工。

如果真的按“很多子 agent 并行”去干，最容易炸的不是请求体，而是下面这三件事：

1. transcript 被 provider 协议消息污染
2. workbench 继续偷偷依赖 Responses 的 item 形态
3. 每个子 agent 都在重新定义“什么叫上下文”

所以这次派工必须遵守一个总原则：

> **所有子 agent 只能改自己那一层，不能顺手改上下文语义。**

最稳妥的执行顺序不是四家一起上，而是：

1. 先把当前 Responses 实现抽成公共核心
2. 只接入 OpenAI Chat Completions，验证 turn-centric 路线
3. 等 transcript / workbench / restore 全部没歪，再接 Claude
4. 最后接 Gemini

也就是说：

- **第一批真正开工的子 agent**：只做到 `Responses 抽象化 + Chat Completions 首接入`
- **Claude / Gemini 子 agent**：先写进派工表，但默认处于 blocked 状态，等前一关验收通过再开

---

## 2. 总控规则

这几个规则是所有子 agent 的共同红线。

### 2.1 transcript 红线

无论底层 provider 怎么组织工具调用，最终 transcript 仍然只能新增：

- 1 条 `user record`
- 1 条 `assistant record`

以下内容**都不能长成新的 transcript 节点**：

- Chat Completions 的 `role: "tool"`
- Claude 的 `tool_result`
- Gemini 的 `functionResponse`
- Responses 的 `function_call_output`

### 2.2 产品主模型红线

产品层最终只认这几类东西：

- `PromptBlock`
- `TranscriptRecord`
- `ToolEvent`
- `CanonicalItem`

以下东西只能活在 adapter 层或 debug 层：

- `tool_calls`
- `tool_use`
- `functionCall`
- `function_call`
- `function_call_output item`

### 2.3 workbench 红线

workbench 以后只能基于 canonical 结构工作，不能继续把 Responses 原始 item 当成主契约。

### 2.4 集成红线

没有通过关卡的波次，下一波不允许开工。

---

## 3. 波次设计

这次不建议按“12 个子 agent 同时启动”来做，而是按 **4 个波次** 推进。

## 波次 0：地基和护栏

目标：

- 先把“不能坏什么”写死
- 先把内部统一数据结构写死

包含子 agent：

- A0 `Contract Guard`
- A1 `Canonical Contract`
- A2 `Runtime Types & BaseAdapter`

## 波次 1：抽 Responses，不改行为

目标：

- 把现在的 Responses 版逻辑搬进 `AgentCore + ResponsesAdapter`
- 功能 100% 不变

包含子 agent：

- A3 `AgentCore Extractor`
- A4 `ResponsesAdapter`
- A5 `Provider Config & Capability`

## 波次 2：首个 turn-centric 入口

目标：

- 只接入 Chat Completions
- 验证 turn-centric 路线在 transcript / workbench / restore 下不歪

包含子 agent：

- A6 `TurnCentricAdapterBase`
- A7 `ChatCompletionsAdapter`
- A8 `Regression Matrix`

## 波次 3：产品层补完与后续 provider

目标：

- 把 workbench、revision、restore、frontend 全部从 Responses 心智迁出来
- 在前一关稳定后，再开 Claude / Gemini

包含子 agent：

- A9 `Context Workbench`
- A10 `Revision / Restore`
- A11 `Frontend Integration`
- A12 `ClaudeAdapter`（blocked）
- A13 `GeminiAdapter`（blocked）
- A14 `Final Integrator / Reviewer`

---

## 4. 子 Agent 详细派工

下面每个子 agent 都写清楚了：

- 职责
- 可改边界
- 禁改边界
- 输入
- 输出
- 验收标准

---

## A0. Contract Guard

### 职责

先把“不能被破坏的行为”固定成 golden tests 和 contract tests。

这个子 agent 不实现新 provider，只负责建护栏。

### 可改文件

- `tests/agent_runtime/*`
- `tests/fixtures/*`
- `docs/runtime_contract.md`

### 禁改文件

- [simple_agent/agent.py](</path/to/hash-code/simple_agent/agent.py>)
- [web_server.py](</path/to/hash-code/web_server.py>)

### 输入

- 当前 Responses 主聊天行为
- 当前 transcript 结构
- 当前 streaming 事件结构

### 输出

- 一组 contract tests
- 一组 golden transcript fixtures
- 一份运行时契约文档

### 验收标准

- 无工具 case：最终 transcript 只新增 `user + assistant`
- 单工具 case：最终 transcript 只新增 `user + assistant`
- 多工具 case：最终 transcript 只新增 `user + assistant`
- 工具事件只出现在 assistant record 内部
- streaming 事件仍只包含前端既有语义：`delta/reset/tool_event/done/error`

### 最容易踩坑

- 只验证 answer，不验证 transcript 结构
- 忽略 `assistant_blocks` 顺序，导致中间文本和工具顺序错乱

---

## A1. Canonical Contract

### 职责

定义 provider 无关的产品内部契约。

### 可改文件

- 新增 `docs/context-canonical-contract.md`
- 新增 `agent_runtime/core/canonical_types.py`
- 新增 `agent_runtime/core/transcript_contract.py`
- 新增 `agent_runtime/core/tool_events.py`
- [react_app/src/types.ts](</path/to/hash-code/react_app/src/types.ts>)

### 禁改文件

- 任何真实 provider 请求逻辑
- `SimpleAgent.run_turn(...)` 主流程

### 输入

- 现有 Responses 数据形态
- 多接口适配实施方案

### 输出

- `PromptBlock`
- `CanonicalItem`
- `TranscriptRecord`
- `ToolEvent`
- `ProviderRaw`
- `AssistantRoundState`

### 验收标准

- Python 和 TS 类型字段名一致
- 不出现 OpenAI/Claude/Gemini 的专属字段作为主字段
- `CanonicalItem.type` 至少支持：
  - `message`
  - `tool_call`
  - `tool_result`
- `provider_raw` 只能是附加调试结构，不参与主逻辑

### 最容易踩坑

- 继续沿用 `providerItems` 作为主心智
- 在 canonical 类型里塞 `role: tool`

---

## A2. Runtime Types & BaseAdapter

### 职责

定义 adapter 的统一接口和运行时事件类型，但不写任何具体 provider。

### 可改文件

- 新增 `agent_runtime/adapters/base.py`
- 新增 `agent_runtime/core/stream_events.py`
- 新增 `agent_runtime/core/prompt_blocks.py`
- 可小改 [simple_agent/tools.py](</path/to/hash-code/simple_agent/tools.py>) 里的工具定义结构

### 禁改文件

- [web_server.py](</path/to/hash-code/web_server.py>)
- 真实 OpenAI / Claude / Gemini 请求实现

### 输入

- A1 的 canonical contract

### 输出

- `BaseAdapter`
- `AdapterStreamEvent`
- `ProviderRequestContext`
- provider-neutral tool schema 基础层

### 验收标准

- BaseAdapter 不 import 任何具体 SDK
- BaseAdapter 不负责工具执行
- 接口能表达：
  - 文本增量
  - 工具调用 ready
  - provider done
  - error

### 最容易踩坑

- 把 BaseAdapter 设计成 OpenAI 专用接口
- 把工具执行逻辑塞进 adapter

---

## A3. AgentCore Extractor

### 职责

把当前 `SimpleAgent` 里的 tool loop 抽成公共核心，但不改变外部行为。

### 可改文件

- 新增 `agent_runtime/core/agent_core.py`
- 小改 [simple_agent/agent.py](</path/to/hash-code/simple_agent/agent.py>)

### 禁改文件

- [web_server.py](</path/to/hash-code/web_server.py>) 的 transcript 落库语义
- workbench 相关逻辑

### 输入

- A2 的 BaseAdapter
- 当前 `SimpleAgent.run_turn(...)` 行为

### 输出

- `AgentCore`
- 复用型 tool loop

### 验收标准

- `SimpleAgent.run_turn(...)` 对外签名不变
- 前端主聊天流式行为不变
- `max_tool_rounds` 超限行为不变
- 工具错误仍以 `tool_event` 进入结果，而不是中断整轮
- 中间轮文本触发 reset 的时机不变

### 最容易踩坑

- 提前更新 `self.history`
- 把中间轮状态直接写进 transcript

---

## A4. ResponsesAdapter

### 职责

把当前 Responses 版请求构造和流式解析搬进 adapter。

### 可改文件

- 新增 `agent_runtime/adapters/responses_adapter.py`
- 小改 [simple_agent/agent.py](</path/to/hash-code/simple_agent/agent.py>)

### 禁改文件

- transcript compiler
- workbench compiler

### 输入

- A3 的 AgentCore
- 当前 Responses 实现

### 输出

- 可替代旧逻辑的 `ResponsesAdapter`

### 验收标准

- 抽象前后，同一输入得到的 `turn_items` 等价
- 抽象前后，assistant `blocks/toolEvents/providerItems` 等价
- 用户无感知
- workbench 原功能不回退

### 最容易踩坑

- 把 Responses 的平级 `function_call` 强行塞回 assistant message
- `call_id` 丢失

---

## A5. Provider Config & Capability

### 职责

建立 provider 配置结构与能力位表。

### 可改文件

- 新增 `agent_runtime/providers/config.py`
- 新增 `agent_runtime/providers/capabilities.py`
- [web_server.py](</path/to/hash-code/web_server.py>)
- [react_app/src/types.ts](</path/to/hash-code/react_app/src/types.ts>)
- [react_app/src/api.ts](</path/to/hash-code/react_app/src/api.ts>)

### 禁改文件

- 具体 adapter 逻辑

### 输入

- provider 列表与能力差异

### 输出

- provider 配置模型
- capability flags

### 建议最少字段

```ts
type ProviderCapabilities = {
  supports_streaming: boolean
  supports_parallel_tool_calls: boolean
  supports_reasoning_summary: boolean
  supports_system_prompt_in_messages: boolean
  supports_top_level_system_prompt: boolean
  supports_tool_calling: boolean
  supports_multimodal_user_input: boolean
  tool_result_role: "tool" | "user" | "function_response_item"
  assistant_tool_call_location: "message_tool_calls" | "content_blocks" | "parts" | "response_items"
}
```

### 验收标准

- UI 能读取当前 provider 和 model
- 后端能读取同一份 provider 配置
- 没有用 `if model.startswith(...)` 这种方式猜能力

### 最容易踩坑

- provider 和 model 混成一个字段
- 前端和后端用两套 capability 逻辑

---

## A6. TurnCentricAdapterBase

### 职责

为 Chat Completions / Claude / Gemini 提供共享骨架。

### 可改文件

- 新增 `agent_runtime/adapters/turn_centric_base.py`

### 禁改文件

- 具体 Chat Completions / Claude / Gemini 请求实现
- transcript 落库逻辑

### 输入

- A1、A2、A3 的成果

### 输出

- turn-centric adapter 抽象层

### 必须抽出来的能力

- prompt blocks 编译入口
- transcript 编译入口
- 当前 round state 注入入口
- tool results 回传入口
- assistant turn assembler

### 验收标准

- 基类不出现 `tool_calls` / `tool_use` / `functionCall` 的业务依赖
- 基类输出仍然是 canonical `message/tool_call/tool_result`

### 最容易踩坑

- 过度抽象，把三家揉成同一个请求模板

---

## A7. ChatCompletionsAdapter

### 职责

作为第一个 turn-centric provider，验证整条路线能跑通。

### 可改文件

- 新增 `agent_runtime/adapters/chat_completions_adapter.py`
- 必要时小改 provider config glue code

### 禁改文件

- workbench 主逻辑
- transcript record shape

### 输入

- A6 的基类
- A5 的 provider config

### 输出

- `ChatCompletionsAdapter`

### 验收标准

- 纯聊天闭环
- 单工具闭环
- 多工具闭环
- 工具报错进入 assistant record
- transcript 不新增 `tool` 节点
- workbench 读取到的 item 结构与 Responses 路径一致

### 最容易踩坑

- `tool_calls[].function.arguments` 在 streaming 下是增量字符串
- `assistant` 可能同时含文本和 `tool_calls`
- Chat Completions tool schema 与 Responses 不同

---

## A8. Regression Matrix

### 职责

建立跨 provider 的回归矩阵和 fake provider 测试。

### 可改文件

- 新增 `tests/test_agent_contract.py`
- 新增 `tests/test_provider_adapters.py`
- 新增 `tests/test_context_workbench.py`
- 新增 `tests/test_revision_restore.py`
- 新增 `tests/fixtures/provider_streams/*`
- 可新增 `scripts/smoke_provider_matrix.py`

### 禁改文件

- 业务核心逻辑

### 输入

- A0 的契约测试
- A4 / A7 的 provider 实现

### 输出

- 可重复跑的 provider 回归矩阵

### 必测矩阵

- 纯聊天
- 单工具
- 多工具
- 工具后继续文本
- 工具错误
- 流式中断
- 有附件
- workbench 详情
- item 删除 / 替换 / 压缩
- revision 提交与恢复
- 切 provider 后旧历史展示

### 验收标准

- Responses 与 Chat Completions 输出的 transcript 结构一致
- provider-specific 字段只出现在 adapter 或 `provider_raw`

### 最容易踩坑

- 只测真实 API
- 只测 answer，不测 transcript 污染

---

## A9. Context Workbench

### 职责

把 workbench 从 Responses 心智迁到 canonical 心智。

### 可改文件

- [web_server.py](</path/to/hash-code/web_server.py>) 中 workbench 相关部分
- 新增 `agent_runtime/workbench/canonical_item_compiler.py`
- 新增 `agent_runtime/workbench/canonical_item_reverse_compiler.py`
- [react_app/src/components/ContextWorkbench.tsx](</path/to/hash-code/react_app/src/components/ContextWorkbench.tsx>)
- [react_app/src/components/ContextMapSidebar.tsx](</path/to/hash-code/react_app/src/components/ContextMapSidebar.tsx>)
- [react_app/src/types.ts](</path/to/hash-code/react_app/src/types.ts>)

### 禁改文件

- provider adapter 具体实现

### 输入

- A1 的 canonical contract
- A4 / A7 的 canonical outputs

### 输出

- provider 无关的 workbench

### 验收标准

- 四家最终都展示成同一种 item 结构
- prompt/memory blocks 在顶部独立显示，不参与 Node 编号
- item 级删除 / 替换 / 压缩不丢工具事件
- reverse compiler 不再是 Responses-only

### 最容易踩坑

- UI 看起来统一了，但 reverse compiler 仍然只懂 Responses

---

## A10. Revision / Restore

### 职责

把版本与恢复语义固定成 provider 无关的快照逻辑。

### 可改文件

- [web_server.py](</path/to/hash-code/web_server.py>) 中 revision/restore 相关部分
- 新增 `agent_runtime/core/revision_snapshot.py`
- 新增 `agent_runtime/core/restore_service.py`
- [react_app/src/components/ContextWorkbench.tsx](</path/to/hash-code/react_app/src/components/ContextWorkbench.tsx>)

### 禁改文件

- adapter 请求体逻辑

### 输入

- A1 / A9 的 canonical transcript 结构

### 输出

- 完整快照式 revision/restore

### 验收标准

- 活版本会持续吸收后续聊天，直到下一版生成才冻结
- 恢复时同时恢复：
  - transcript
  - context workbench history
  - context model chat history
  - prompt/memory blocks
- 不再出现“撤销此次切换”这类误导语义

### 最容易踩坑

- restore 只恢复 transcript，不恢复 workbench chat

---

## A11. Frontend Integration

### 职责

做 provider 选择、能力提示与页面交互收口。

### 可改文件

- [react_app/src/components/SettingsView.tsx](</path/to/hash-code/react_app/src/components/SettingsView.tsx>)
- [react_app/src/components/ContextWorkbench.tsx](</path/to/hash-code/react_app/src/components/ContextWorkbench.tsx>)
- [react_app/src/api.ts](</path/to/hash-code/react_app/src/api.ts>)
- [react_app/src/types.ts](</path/to/hash-code/react_app/src/types.ts>)
- [react_app/src/react-entry.css](</path/to/hash-code/react_app/src/react-entry.css>)
- [static/css/style.css](</path/to/hash-code/static/css/style.css>)

### 禁改文件

- 核心 transcript 语义

### 输入

- A5 的 provider config/capabilities
- A9/A10 的后端接口

### 输出

- 用户可见的 provider 切换与能力反馈

### 验收标准

- 用户能清楚看到当前 provider
- 不支持的能力不会显示成可点击但必失败
- workbench / restore / settings 页面文案和状态一致

### 最容易踩坑

- 前端自己猜 provider 能力，而不是读后端 flags

---

## A12. ClaudeAdapter（Blocked）

### 开工前提

只有在下面这几个条件同时满足后才允许开工：

1. A4 通过
2. A7 通过
3. A8 回归矩阵通过 Responses + Chat Completions
4. A9 / A10 没有 provider 污染问题

### 职责

接入 Claude Messages，不改变产品语义。

### 验收标准

- `tool_result` 不会变成 user transcript 节点
- assistant record 结构与前两家一致

---

## A13. GeminiAdapter（Blocked）

### 开工前提

与 A12 相同，并且 Claude 接入后没有新的 transcript 污染。

### 职责

接入 Gemini GenerateContent，不改变产品语义。

### 验收标准

- `functionResponse` 不会变成 user transcript 节点
- assistant record 结构与前几家一致

---

## A14. Final Integrator / Reviewer

### 职责

最后统一把关，只修集成缝隙，不重写业务逻辑。

### 可改文件

- 少量 glue code
- 测试
- adapter factory
- 文档

### 禁改文件

- 不许为了让测试过而降低 transcript 契约

### 最终验收清单

- Responses 抽象化后老功能无回归
- Chat Completions 接入后主聊天能跑多轮工具
- transcript record shape 一致
- workbench item shape 一致
- `append_turn` 仍然只追加两条产品层记录
- 前端不需要知道 provider 协议细节
- 切 provider 后旧历史仍可展示
- Claude / Gemini 只有在前面路线完全稳定后才启动

---

## 5. 推荐执行顺序

### 第一批立刻开工

1. A0 `Contract Guard`
2. A1 `Canonical Contract`
3. A2 `Runtime Types & BaseAdapter`

### 第二批

4. A3 `AgentCore Extractor`
5. A4 `ResponsesAdapter`
6. A5 `Provider Config & Capability`

### 第一个关卡 G1

G1 必须同时满足：

- Responses 行为零回归
- contract tests 全绿
- transcript 没被污染

没过 G1，A6 以后全部停住。

### 第三批

7. A6 `TurnCentricAdapterBase`
8. A7 `ChatCompletionsAdapter`
9. A8 `Regression Matrix`

### 第二个关卡 G2

G2 必须同时满足：

- Chat Completions 多轮工具闭环通过
- Responses / Chat Completions transcript shape 一致
- workbench 能读两家输出

没过 G2，Claude / Gemini 不许开工。

### 第四批

10. A9 `Context Workbench`
11. A10 `Revision / Restore`
12. A11 `Frontend Integration`

### 第五批（blocked 转 runnable）

13. A12 `ClaudeAdapter`
14. A13 `GeminiAdapter`

### 最终关卡 G3

15. A14 `Final Integrator / Reviewer`

---

## 6. 最后我怎么统一把关

如果真按这个方案派出去，最后我不会只看“能不能回复”，我会按下面这张表做总审。

### 6.1 transcript 审核

- 有没有新增 provider 协议角色进入 transcript
- 有没有把工具结果错误变成 user 节点
- assistant record 是否仍是单轮聚合结果

### 6.2 workbench 审核

- 详情页是否只读 canonical item
- reverse compiler 是否已脱离 Responses-only
- item 编辑后 `blocks/toolEvents/text/canonicalItems` 是否一致

### 6.3 revision 审核

- 活版本是否持续吸收聊天直到下一版生成
- restore 是否恢复完整快照而不是单点状态

### 6.4 provider 审核

- provider-specific 逻辑是否只出现在 adapter 层
- 有没有地方还在偷偷用 `if provider == ...` 写业务语义分叉

### 6.5 回归审核

- Responses 老功能是否零回退
- Chat Completions 是否只是“能聊”，还是整套产品功能都没歪
- Claude / Gemini 是否在 blocked 规则满足后才开工

---

## 7. 最现实的建议

如果你真准备这么干，我建议第一轮**只派到这里**：

1. A0
2. A1
3. A2
4. A3
5. A4
6. A5
7. A6
8. A7
9. A8
10. A14

也就是：

> **先做到 Responses 抽象化 + Chat Completions 首接入 + 全套护栏。**

Claude 和 Gemini 先写进计划，但先别真开工。  
因为如果 Chat Completions 这关都还没把 transcript/workbench/revision 的语义走通，后面再多两个 provider，只会让问题更难查。
