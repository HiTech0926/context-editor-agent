from __future__ import annotations

import json

from web_server import provider_input_to_context_records


def test_provider_input_visualization_preserves_non_assistant_order() -> None:
    input_items = [
        {
            "type": "message",
            "role": "developer",
            "content": "developer rules",
        },
        {
            "type": "message",
            "role": "user",
            "content": "environment context",
        },
        {
            "type": "message",
            "role": "user",
            "content": "latest user turn",
        },
        {
            "type": "compaction",
            "summary": "older conversation summary",
        },
    ]

    records = provider_input_to_context_records(input_items)

    assert [record["role"] for record in records] == [
        "developer",
        "user",
        "user",
        "compaction",
    ]
    assert [record["text"] for record in records] == [
        "developer rules",
        "environment context",
        "latest user turn",
        "older conversation summary",
    ]
    assert [record["providerItems"][0]["type"] for record in records] == [
        "message",
        "message",
        "message",
        "compaction",
    ]


def test_provider_input_visualization_folds_assistant_protocol_items() -> None:
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": "inspect files",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": "I will inspect the tree.",
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "shell_command",
            "arguments": json.dumps({"command": "rg --files"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "README.md\nweb_server.py",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": "The project has Python and React files.",
        },
    ]

    records = provider_input_to_context_records(input_items)

    assert [record["role"] for record in records] == ["user", "assistant"]
    assistant_record = records[1]
    assert assistant_record["text"] == "I will inspect the tree.The project has Python and React files."
    assert [item["type"] for item in assistant_record["providerItems"]] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert len(assistant_record["toolEvents"]) == 1
