from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_runtime.core.agent_core import AgentCore


@dataclass(slots=True)
class _Call:
    name: str
    arguments: str
    call_id: str


@dataclass(slots=True)
class _Response:
    output_text: str
    function_calls: list[_Call]
    canonical_items: tuple[dict[str, Any], ...] = ()


@dataclass(slots=True)
class _Execution:
    output_text: str = "tool output"
    display_title: str = "tool"
    display_detail: str = ""
    display_result: str = ""
    status: str = "completed"


def test_agent_core_keeps_provider_assistant_item_before_tool_result() -> None:
    history: list[dict[str, Any]] = []
    calls = [_Call(name="read_file", arguments='{"path":"README.md"}', call_id="toolu_1")]
    provider_item = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Need a file.", "signature": "sig"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read_file",
                "input": {"path": "README.md"},
            },
        ],
    }
    responses = iter(
        [
            _Response(output_text="", function_calls=calls, canonical_items=(provider_item,)),
            _Response(output_text="done", function_calls=[]),
        ]
    )
    captured_turns: list[list[dict[str, Any]]] = []

    def build_request(
        turn_items: list[dict[str, Any]],
        request_model: str,
        request_reasoning_effort: str | None,
    ) -> dict[str, Any]:
        del request_model, request_reasoning_effort
        captured_turns.append([dict(item) for item in turn_items])
        return {}

    core = AgentCore(
        max_tool_rounds=3,
        default_model="model",
        history=history,
        build_request=build_request,
        stream_response=lambda **_: next(responses),
        execute_tool=lambda _name, _arguments: _Execution(),
        make_user_message=lambda text, attachments=None: {
            "type": "message",
            "role": "user",
            "content": text,
        },
        make_assistant_message=lambda text: {
            "type": "message",
            "role": "assistant",
            "content": text,
        },
        tool_event_factory=lambda **kwargs: kwargs,
        sanitize_text=str,
        sanitize_value=lambda value: value,
        preview_text=lambda text: text,
        should_fallback_to_developer=lambda exc: False,
        fallback_to_developer=lambda: None,
    )

    answer, _tool_events = core.run_turn("hello")

    assert answer == "done"
    assert captured_turns[1][1] == provider_item
    assert captured_turns[1][2] == {
        "type": "function_call_output",
        "call_id": "toolu_1",
        "output": "tool output",
    }
    assert history[1] == provider_item
