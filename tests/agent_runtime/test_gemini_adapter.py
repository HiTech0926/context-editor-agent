from __future__ import annotations

from agent_runtime.adapters.base import ProviderRequestContext
from agent_runtime.adapters.gemini_adapter import GeminiAdapter
from agent_runtime.core.stream_events import ProviderDoneEvent, TextDeltaEvent


class EmptyStreamFallbackClient:
    def __init__(self) -> None:
        self.generate_content_calls = 0

    def stream_generate_content(self, **request: object) -> list[dict[str, object]]:
        return []

    def generate_content(self, **request: object) -> dict[str, object]:
        self.generate_content_calls += 1
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "fallback text"}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }


def test_build_request_parses_json_string_tool_results_into_object() -> None:
    adapter = GeminiAdapter(client=object())
    request = adapter.build_request(
        ProviderRequestContext(
            model="gemini-2.5-pro",
            current_turn=(
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "{\"content\":\"hello\",\"size\":5}",
                },
            ),
        )
    )

    function_response = request["contents"][1]["parts"][0]["functionResponse"]
    assert function_response["name"] == "read_file"
    assert function_response["response"] == {"content": "hello", "size": 5}


def test_build_request_wraps_plain_text_tool_results_for_gemini() -> None:
    adapter = GeminiAdapter(client=object())
    request = adapter.build_request(
        ProviderRequestContext(
            model="gemini-2.5-pro",
            current_turn=(
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "echo",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "plain text result",
                },
            ),
        )
    )

    function_response = request["contents"][1]["parts"][0]["functionResponse"]
    assert function_response["name"] == "echo"
    assert function_response["response"] == {"result": "plain text result"}


def test_stream_response_falls_back_to_generate_content_when_stream_is_empty() -> None:
    client = EmptyStreamFallbackClient()
    adapter = GeminiAdapter(client=client)
    request = adapter.build_request(
        ProviderRequestContext(
            model="gemini-2.5-pro",
            current_turn=(
                {
                    "type": "message",
                    "role": "user",
                    "content": "hello",
                },
            ),
        )
    )

    events = list(adapter.stream_response(request))

    assert client.generate_content_calls == 1
    assert any(
        isinstance(event, TextDeltaEvent) and event.delta == "fallback text"
        for event in events
    )
    done_events = [event for event in events if isinstance(event, ProviderDoneEvent)]
    assert done_events[-1].output_text == "fallback text"
    assert done_events[-1].finish_reason == "STOP"
