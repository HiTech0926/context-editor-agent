from __future__ import annotations

import json

from web_server_modules.context_workbench import ContextWorkbenchDraft, ContextWorkbenchToolRegistry


def make_transcript() -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "text": "Please analyze the failing import.",
            "attachments": [],
            "toolEvents": [],
            "blocks": [{"kind": "text", "text": "Please analyze the failing import."}],
            "providerItems": [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Please analyze the failing import.",
                }
            ],
        },
        {
            "role": "assistant",
            "text": "Long tool output that should be summarized.",
            "attachments": [],
            "toolEvents": [],
            "blocks": [{"kind": "text", "text": "Long tool output that should be summarized."}],
            "providerItems": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Long tool output that should be summarized.",
                }
            ],
        },
    ]


def decode_tool_output(output_text: str) -> dict[str, object]:
    decoded = json.loads(output_text)
    assert isinstance(decoded, dict)
    return decoded


def test_tool_catalog_removes_preview_and_adds_final_confirmation() -> None:
    tool_ids = {item["id"] for item in ContextWorkbenchToolRegistry.tool_catalog()}

    assert "preview_context_selection" not in tool_ids
    assert "confirm_working_snapshot" in tool_ids


def test_tools_do_not_expose_hint_based_targeting() -> None:
    registry = ContextWorkbenchToolRegistry(ContextWorkbenchDraft(make_transcript(), selected_indexes=[]))
    forbidden_hint_param = "target" + "_hint"

    for schema in registry.schemas:
        parameters = schema.get("parameters") or {}
        properties = parameters.get("properties") or {}
        assert forbidden_hint_param not in properties

    schemas_by_name = {schema["name"]: schema for schema in registry.schemas}
    assert schemas_by_name["confirm_working_snapshot"]["parameters"]["required"] == []

    mutation_tools = [
        "get_context_node_details",
        "delete_context_item",
        "replace_context_item",
        "compress_context_item",
        "compress_context_nodes",
        "delete_context_nodes",
    ]
    for tool_id in mutation_tools:
        assert "node_numbers" in schemas_by_name[tool_id]["parameters"]["required"]


def test_mutation_without_node_numbers_requests_explicit_target() -> None:
    draft = ContextWorkbenchDraft(make_transcript(), selected_indexes=[1])
    registry = ContextWorkbenchToolRegistry(draft)

    result = decode_tool_output(
        registry.execute(
            "compress_context_nodes",
            {"summary_markdown": "Summarized import failure analysis."},
        ).output_text
    )

    assert result["payload_kind"] == "target_resolution"
    assert result["resolved"] is False
    assert result["selected_node_numbers"] == [2]
    assert result["candidates"] == []
    assert draft.final_snapshot_payload()["active_node_count"] == 2


def test_mutation_returns_delta_and_confirmation_returns_final_snapshot() -> None:
    draft = ContextWorkbenchDraft(make_transcript(), selected_indexes=[])
    registry = ContextWorkbenchToolRegistry(draft)

    mutation = decode_tool_output(
        registry.execute(
            "compress_context_nodes",
            {
                "node_numbers": [2],
                "summary_markdown": "Summarized import failure analysis.",
            },
        ).output_text
    )

    assert mutation["payload_kind"] == "mutation_delta"
    assert "working_overview" not in mutation
    assert mutation["compressed_node_numbers"] == [2]
    assert mutation["created_node"]["label"] == "Draft Node A"

    confirmation = decode_tool_output(registry.execute("confirm_working_snapshot", {}).output_text)

    assert confirmation["payload_kind"] == "final_working_snapshot"
    assert confirmation["active_node_count"] == 2
    assert confirmation["inactive_node_count"] == 1
    assert [node["label"] for node in confirmation["active_nodes"]] == ["Node #1", "Draft Node A"]
    assert confirmation["inactive_nodes"] == [
        {
            "node_number": 2,
            "label": "Node #2",
            "status": "compressed",
            "node_kind": "existing",
            "active": False,
            "replaced_by": "Draft Node A",
        }
    ]
