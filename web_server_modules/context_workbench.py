from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency fallback for partially installed environments
    tiktoken = None

from simple_agent.agent import sanitize_text
from simple_agent.tools import ToolExecution

from .attachments import normalize_attachment_records
from .serialization import sanitize_value
from .transcript import (
    block_text_preview,
    compile_record_from_provider_items,
    context_detail_block,
    extract_text_from_provider_message_content,
    extract_tool_events_from_blocks,
    normalize_message_blocks,
    normalize_provider_items,
    normalize_transcript,
    provider_item_detail,
    replace_provider_message_text,
)


_TOKEN_ENCODING: Any | None = None
_TOKEN_ENCODING_LOAD_FAILED = False


@dataclass(slots=True)
class ContextWorkbenchToolDefinition:
    name: str
    label: str
    description: str
    parameters: dict[str, Any]
    status: str
    handler: Callable[[dict[str, Any]], ToolExecution]

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_catalog_item(self) -> dict[str, str]:
        return {
            "id": self.name,
            "label": self.label,
            "description": self.description,
            "status": self.status,
        }


def provider_items_tool_token_count(items: list[dict[str, Any]]) -> int:
    total = 0
    for item in items:
        if sanitize_text(item.get("type") or "").strip() not in {"function_call", "function_call_output"}:
            continue
        total += estimate_provider_item_token_count(item)
    return total


def context_workbench_suggestions_payload(session: SessionState) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    for index, record in enumerate(session.transcript):
        overview = context_record_overview(record, node_number=index + 1)
        token_count = int(overview.get("token_estimate") or 0)
        tool_token_count = int(overview.get("tool_token_estimate") or 0)
        nodes.append(
            {
                "node_index": index,
                "node_number": index + 1,
                "role": sanitize_text(overview.get("role") or "").strip() or "assistant",
                "token_count": token_count,
                "tool_token_count": tool_token_count,
                "preview": sanitize_text(overview.get("preview") or "").strip(),
            }
        )

    nodes.sort(
        key=lambda item: (
            -int(item.get("token_count") or 0),
            int(item.get("node_number") or 0),
        )
    )

    return {
        "stats": {
            "total_token_count": sum(int(item.get("token_count") or 0) for item in nodes),
            "tool_token_count": sum(int(item.get("tool_token_count") or 0) for item in nodes),
        },
        "nodes": sanitize_value(nodes),
    }


def normalize_selected_node_indexes(raw_indexes: Any, transcript_length: int) -> list[int]:
    if not isinstance(raw_indexes, list):
        return []

    selected_indexes: list[int] = []
    for raw_item in raw_indexes:
        try:
            index = int(raw_item)
        except (TypeError, ValueError):
            continue

        if 0 <= index < transcript_length and index not in selected_indexes:
            selected_indexes.append(index)

    return selected_indexes


def normalize_node_numbers(raw_numbers: Any, max_node_number: int) -> list[int]:
    if not isinstance(raw_numbers, list):
        return []

    normalized: list[int] = []
    for raw_item in raw_numbers:
        try:
            node_number = int(raw_item)
        except (TypeError, ValueError):
            continue

        if 1 <= node_number <= max_node_number and node_number not in normalized:
            normalized.append(node_number)

    return normalized


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_token_encoding() -> Any | None:
    global _TOKEN_ENCODING, _TOKEN_ENCODING_LOAD_FAILED

    if _TOKEN_ENCODING is not None:
        return _TOKEN_ENCODING
    if _TOKEN_ENCODING_LOAD_FAILED or tiktoken is None:
        return None

    try:
        _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKEN_ENCODING_LOAD_FAILED = True
        return None

    return _TOKEN_ENCODING


def estimate_token_count(text: str) -> int:
    safe_text = sanitize_text(text)
    if not safe_text.strip():
        return 0

    encoding = get_token_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(safe_text))
        except Exception:
            pass

    compact = safe_text.strip()
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", compact)
    non_ascii_chars = [char for char in compact if not char.isspace() and not char.isascii()]
    return max(1, len(ascii_tokens) + len(non_ascii_chars))


def unique_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []

    unique_values: list[int] = []
    for raw_value in values:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def unique_text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    unique_values: list[str] = []
    for raw_value in values:
        value = sanitize_text(raw_value or "").strip()
        if value and value not in unique_values:
            unique_values.append(value)
    return unique_values


def operation_changed_nodes(operation: dict[str, object]) -> list[int]:
    explicit_nodes = unique_int_list(operation.get("changed_nodes"))
    if explicit_nodes:
        return explicit_nodes

    target_nodes = unique_int_list(operation.get("target_node_numbers"))
    if target_nodes:
        return target_nodes

    target_items = operation.get("target_items")
    if isinstance(target_items, list):
        item_nodes: list[int] = []
        for item in target_items:
            if not isinstance(item, dict):
                continue
            try:
                node_number = int(item.get("node_number") or 0)
            except (TypeError, ValueError):
                continue
            if node_number > 0 and node_number not in item_nodes:
                item_nodes.append(node_number)
        if item_nodes:
            return item_nodes

    return []


def normalize_change_type(raw_value: Any) -> str:
    value = sanitize_text(raw_value or "").strip().lower()
    if value in {"delete", "replace", "compress", "mixed", "update"}:
        return value
    if value.startswith("delete"):
        return "delete"
    if value.startswith("replace"):
        return "replace"
    if value.startswith("compress"):
        return "compress"
    return "update"


def operation_change_type(operation: dict[str, object]) -> str:
    return normalize_change_type(
        operation.get("change_type")
        or operation.get("operation_type")
        or operation.get("type")
        or "update"
    )


def summarize_change_type(change_types: list[str]) -> str:
    normalized = [normalize_change_type(item) for item in change_types if sanitize_text(item).strip()]
    unique_types = [item for item in normalized if item]
    if not unique_types:
        return "update"
    if len(set(unique_types)) == 1:
        return unique_types[0]
    return "mixed"


def summarize_changed_nodes_from_operations(operations: list[dict[str, object]]) -> list[int]:
    changed_nodes: list[int] = []
    for operation in operations:
        for node_number in operation_changed_nodes(operation):
            if node_number not in changed_nodes:
                changed_nodes.append(node_number)
    return changed_nodes


def fallback_context_revision_summary(label: str, operations: list[dict[str, object]]) -> str:
    safe_label = sanitize_text(label).strip() or "Context update"
    if not operations:
        return safe_label

    if len(operations) == 1:
        operation = operations[0]
        operation_type = sanitize_text(operation.get("operation_type") or "").strip()
        target_nodes = unique_int_list(operation.get("target_node_numbers") or operation.get("changed_nodes"))
        node_text = f"节点 #{format_node_ranges(target_nodes)}" if target_nodes else "当前上下文"
        target_items = operation.get("target_items")
        first_item = target_items[0] if isinstance(target_items, list) and target_items else {}
        item_number = int(first_item.get("item_number") or 0) if isinstance(first_item, dict) else 0
        item_text = f"{node_text} 的第 {item_number} 个条目" if item_number else node_text

        if operation_type == "compress_nodes":
            return f"把{node_text}压缩成了更短的摘要，尽量保留主要信息。"
        if operation_type == "delete_nodes":
            return f"删除了{node_text}，让当前上下文更紧凑。"
        if operation_type == "delete_item":
            return f"删除了{item_text}，去掉了不再需要的上下文内容。"
        if operation_type == "compress_item":
            return f"压缩了{item_text}，保留原有条目类型的同时缩短了内容。"
        if operation_type == "replace_item":
            return f"改写了{item_text}，把它换成了更合适的新内容。"

    changed_nodes = summarize_changed_nodes_from_operations(operations)
    if changed_nodes:
        return f"这一轮集中更新了节点 #{format_node_ranges(changed_nodes)} 的内容，并把它们整理成了新的上下文版本。"
    return safe_label


def find_active_context_revision_id(revisions: list[dict[str, object]]) -> str | None:
    for revision in revisions:
        revision_id = sanitize_text(revision.get("id") or "").strip()
        if revision_id and bool(revision.get("is_active")):
            return revision_id
    return None


def mark_active_context_revision(revisions: list[dict[str, object]], revision_id: str | None) -> None:
    safe_revision_id = sanitize_text(revision_id or "").strip()
    for revision in revisions:
        current_id = sanitize_text(revision.get("id") or "").strip()
        revision["is_active"] = bool(safe_revision_id and current_id == safe_revision_id)


def coerce_context_revision_number(raw_value: Any, fallback: int, *, minimum: int = 0) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = int(fallback)
    return max(minimum, value)


def has_initial_context_revision(revisions: list[dict[str, object]]) -> bool:
    return any(
        coerce_context_revision_number(revision.get("revision_number"), 1) == 0
        for revision in revisions
    )


def next_context_revision_number(revisions: list[dict[str, object]]) -> int:
    numbers = [
        coerce_context_revision_number(revision.get("revision_number"), 0)
        for revision in revisions
    ]
    return max([number for number in numbers if number > 0], default=0) + 1


def ensure_initial_context_revision(session: SessionState) -> None:
    if has_initial_context_revision(session.context_revisions):
        return
    if session.context_revisions:
        return

    session.context_revisions.append(
        build_context_revision_entry(
            transcript=normalize_transcript(session.transcript),
            context_workbench_history=normalize_context_chat_history(session.context_workbench_history),
            revision_label="初始版本",
            revision_summary="还没有进行压缩、删除或替换时的完整上下文。",
            operations=[],
            revision_number=0,
        )
    )


def sync_active_context_revision_snapshot(session: SessionState) -> None:
    active_revision_id = find_active_context_revision_id(session.context_revisions)
    if not active_revision_id:
        return

    safe_snapshot = sanitize_value(normalize_transcript(session.transcript))
    safe_context_workbench_history = sanitize_value(
        normalize_context_chat_history(session.context_workbench_history)
    )
    for revision in reversed(session.context_revisions):
        current_id = sanitize_text(revision.get("id") or "").strip()
        if current_id != active_revision_id:
            continue
        revision["snapshot"] = safe_snapshot
        revision["context_workbench_history_snapshot"] = safe_context_workbench_history
        revision["node_count"] = len(session.transcript)
        return


def build_context_revision_entry(
    *,
    transcript: list[dict[str, object]],
    context_workbench_history: list[dict[str, str]],
    revision_label: str,
    revision_summary: str,
    operations: list[dict[str, object]],
    revision_number: int,
) -> dict[str, object]:
    sanitized_operations = [
        sanitize_value(operation)
        for operation in operations
        if isinstance(operation, dict)
    ]
    changed_nodes = summarize_changed_nodes_from_operations(sanitized_operations)
    change_types = [
        operation_change_type(operation)
        for operation in sanitized_operations
    ]
    label = sanitize_text(revision_label).strip() or "Context update"
    summary = sanitize_text(revision_summary).strip() or fallback_context_revision_summary(label, sanitized_operations)
    return {
        "id": uuid.uuid4().hex,
        "label": label,
        "summary": summary,
        "created_at": utc_timestamp(),
        "revision_number": coerce_context_revision_number(revision_number, 1),
        "change_type": summarize_change_type(change_types),
        "change_types": unique_text_list(change_types),
        "changed_nodes": changed_nodes,
        "operations": sanitized_operations,
        "node_count": len(transcript),
        "snapshot": sanitize_value(transcript),
        "context_workbench_history_snapshot": sanitize_value(
            normalize_context_chat_history(context_workbench_history)
        ),
        "is_active": True,
    }


def normalize_context_revision_entries(raw_entries: Any) -> list[dict[str, object]]:
    if not isinstance(raw_entries, list):
        return []

    normalized: list[dict[str, object]] = []
    for index, item in enumerate(raw_entries, start=1):
        if not isinstance(item, dict):
            continue

        revision_id = sanitize_text(item.get("id") or "").strip()
        label = sanitize_text(item.get("label") or "").strip()
        created_at = sanitize_text(item.get("created_at") or "").strip() or utc_timestamp()
        snapshot = normalize_transcript(item.get("snapshot"))
        context_workbench_history_snapshot = normalize_context_chat_history(
            item.get("context_workbench_history_snapshot")
        )
        operations = sanitize_value(item.get("operations")) if isinstance(item.get("operations"), list) else []
        if not revision_id or not label:
            continue

        changed_nodes = unique_int_list(item.get("changed_nodes")) or summarize_changed_nodes_from_operations(operations)
        change_types = unique_text_list(item.get("change_types"))
        if not change_types:
            change_types = [operation_change_type(operation) for operation in operations if isinstance(operation, dict)]
        change_type = normalize_change_type(item.get("change_type") or summarize_change_type(change_types))

        summary = sanitize_text(item.get("summary") or "").strip()
        if not summary or summary == label:
            summary = fallback_context_revision_summary(label, operations)

        normalized.append(
            {
                "id": revision_id,
                "label": label,
                "summary": summary,
                "created_at": created_at,
                "revision_number": coerce_context_revision_number(
                    item.get("revision_number"),
                    index,
                ),
                "change_type": change_type,
                "change_types": unique_text_list(change_types) or [change_type],
                "changed_nodes": changed_nodes,
                "operations": operations,
                "node_count": len(snapshot),
                "snapshot": sanitize_value(snapshot),
                "context_workbench_history_snapshot": sanitize_value(context_workbench_history_snapshot),
                "is_active": bool(item.get("is_active")),
            }
        )

    if normalized and not any(bool(revision.get("is_active")) for revision in normalized):
        normalized[-1]["is_active"] = True

    for revision_number, revision in enumerate(normalized, start=1):
        revision["revision_number"] = coerce_context_revision_number(
            revision.get("revision_number"),
            revision_number,
        )

    return normalized


def normalize_pending_context_restore(raw_restore: Any) -> dict[str, object] | None:
    if not isinstance(raw_restore, dict):
        return None

    undo_transcript = normalize_transcript(raw_restore.get("undo_transcript"))
    undo_context_workbench_history = normalize_context_chat_history(
        raw_restore.get("undo_context_workbench_history")
    )
    target_revision_id = sanitize_text(raw_restore.get("target_revision_id") or "").strip()
    target_label = sanitize_text(raw_restore.get("target_label") or "").strip()
    created_at = sanitize_text(raw_restore.get("created_at") or "").strip() or utc_timestamp()
    undo_active_revision_id = sanitize_text(raw_restore.get("undo_active_revision_id") or "").strip()
    if not undo_transcript or not target_revision_id:
        return None

    return {
        "undo_transcript": sanitize_value(undo_transcript),
        "undo_context_workbench_history": sanitize_value(undo_context_workbench_history),
        "target_revision_id": target_revision_id,
        "target_label": target_label or "Revision",
        "created_at": created_at,
        "undo_active_revision_id": undo_active_revision_id,
    }


def context_revision_summaries(revisions: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": sanitize_text(revision.get("id") or "").strip(),
            "label": sanitize_text(revision.get("label") or "").strip() or "Revision",
            "summary": (
                lambda label, summary, operations: (
                    fallback_context_revision_summary(label, operations)
                    if not summary or summary == label
                    else summary
                )
            )(
                sanitize_text(revision.get("label") or "").strip() or "Revision",
                sanitize_text(revision.get("summary") or "").strip(),
                sanitize_value(revision.get("operations")) if isinstance(revision.get("operations"), list) else [],
            ),
            "created_at": sanitize_text(revision.get("created_at") or "").strip() or utc_timestamp(),
            "revision_number": coerce_context_revision_number(revision.get("revision_number"), 0),
            "change_type": normalize_change_type(revision.get("change_type") or "update"),
            "change_types": unique_text_list(revision.get("change_types")) or [
                normalize_change_type(revision.get("change_type") or "update")
            ],
            "changed_nodes": unique_int_list(revision.get("changed_nodes")),
            "is_active": bool(revision.get("is_active")),
            "operation_count": len(revision.get("operations") or []),
            "node_count": int(revision.get("node_count") or 0),
        }
        for revision in reversed(revisions)
        if sanitize_text(revision.get("id") or "").strip()
    ]


def context_pending_restore_payload(raw_restore: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(raw_restore, dict):
        return None

    target_revision_id = sanitize_text(raw_restore.get("target_revision_id") or "").strip()
    if not target_revision_id:
        return None

    return {
        "target_revision_id": target_revision_id,
        "target_label": sanitize_text(raw_restore.get("target_label") or "").strip() or "Revision",
        "created_at": sanitize_text(raw_restore.get("created_at") or "").strip() or utc_timestamp(),
        "undo_active_revision_id": sanitize_text(raw_restore.get("undo_active_revision_id") or "").strip(),
        "can_undo": True,
    }


def context_record_preview(record: dict[str, object], *, limit: int = 140) -> str:
    blocks = normalize_message_blocks(record.get("blocks"))
    attachments = normalize_attachment_records(record.get("attachments"))
    text = sanitize_text(record.get("text") or "")

    if blocks:
        for block in blocks:
            kind = sanitize_text(block.get("kind") or "").strip()
            if kind == "text":
                preview = block_text_preview(block.get("text") or "", limit=limit)
                if preview:
                    return preview
                continue

            if kind != "tool":
                continue

            tool_event = block.get("tool_event")
            if not isinstance(tool_event, dict):
                continue
            tool_name = sanitize_text(tool_event.get("name") or tool_event.get("display_title") or "").strip() or "tool"
            tool_detail = block_text_preview(tool_event.get("display_detail") or "", limit=max(40, min(limit, 88)))
            if tool_detail:
                return f"{tool_name}: {tool_detail}"
            return tool_name

    if text:
        return block_text_preview(text, limit=limit)

    if attachments:
        attachment_names = ", ".join(
            sanitize_text(item.get("name") or "").strip()
            for item in attachments
            if sanitize_text(item.get("name") or "").strip()
        )
        if attachment_names:
            return f"Attachments: {attachment_names}"

    return "[empty]"


def record_tool_usage(record: dict[str, object]) -> list[dict[str, object]]:
    tool_events = sanitize_value(record.get("toolEvents")) if isinstance(record.get("toolEvents"), list) else []
    if not tool_events:
        tool_events = extract_tool_events_from_blocks(normalize_message_blocks(record.get("blocks")))

    counts: dict[str, int] = {}
    for tool_event in tool_events:
        if not isinstance(tool_event, dict):
            continue
        tool_name = sanitize_text(tool_event.get("name") or tool_event.get("display_title") or "").strip() or "tool"
        counts[tool_name] = counts.get(tool_name, 0) + 1

    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def format_tool_usage(tool_usage: list[dict[str, object]]) -> str:
    if not tool_usage:
        return "none"

    return ", ".join(
        f"{sanitize_text(item.get('name') or '').strip() or 'tool'} x{int(item.get('count') or 0)}"
        for item in tool_usage
    )


def format_token_count(token_estimate: int) -> str:
    safe_value = max(0, int(token_estimate or 0))
    if safe_value >= 1000:
        return f"{safe_value / 1000:.1f}k"
    return str(safe_value)


def record_context_tool_weight_source(record: dict[str, object]) -> str:
    parts: list[str] = []
    for block in normalize_message_blocks(record.get("blocks")):
        kind = sanitize_text(block.get("kind") or "").strip()
        if kind != "tool":
            continue

        tool_event = block.get("tool_event")
        if not isinstance(tool_event, dict):
            continue

        tool_parts = [
            sanitize_text(tool_event.get("display_title") or "").strip(),
            sanitize_text(tool_event.get("display_detail") or "").strip(),
            sanitize_text(tool_event.get("output_preview") or "").strip(),
            sanitize_text(tool_event.get("display_result") or "").strip(),
            sanitize_text(tool_event.get("raw_output") or "").strip(),
        ]
        joined = "\n".join(part for part in tool_parts if part)
        if joined:
            parts.append(joined)

    return "\n\n".join(parts)


def record_context_weight_source(record: dict[str, object]) -> str:
    parts: list[str] = []
    for block in normalize_message_blocks(record.get("blocks")):
        kind = sanitize_text(block.get("kind") or "").strip()
        if kind == "text":
            text = sanitize_text(block.get("text") or "")
            if text.strip():
                parts.append(text)
            continue

        if kind in {"reasoning", "thinking"}:
            continue

        tool_event = block.get("tool_event")
        if not isinstance(tool_event, dict):
            continue

        tool_source = record_context_tool_weight_source({"blocks": [block]})
        if tool_source:
            parts.append(tool_source)

    if not parts:
        text = sanitize_text(record.get("text") or "")
        if text.strip():
            parts.append(text)

    raw_attachments = record.get("attachments")
    attachments = raw_attachments if isinstance(raw_attachments, list) else []
    attachment_names = "\n".join(
        sanitize_text(attachment.get("name") or "").strip()
        for attachment in attachments
        if isinstance(attachment, dict) and sanitize_text(attachment.get("name") or "").strip()
    )
    if attachment_names:
        parts.append(attachment_names)

    return "\n\n".join(part for part in parts if part.strip())


def context_record_overview(record: dict[str, object], *, node_number: int, selected: bool = False) -> dict[str, object]:
    role = sanitize_text(record.get("role") or "").strip() or "unknown"
    preview = context_record_preview(record)
    tool_usage = record_tool_usage(record)
    provider_items = normalize_provider_items(record.get("providerItems"))
    token_estimate = estimate_token_count(record_context_weight_source(record))
    tool_token_estimate = estimate_token_count(record_context_tool_weight_source(record))
    return {
        "node_number": node_number,
        "role": role,
        "selected": selected,
        "preview": preview,
        "token_estimate": token_estimate,
        "tool_token_estimate": tool_token_estimate,
        "tool_usage": tool_usage,
        "tool_count": sum(int(item.get("count") or 0) for item in tool_usage),
        "item_count": len(provider_items),
        "item_types": [
            sanitize_text(item.get("type") or "").strip() or "unknown"
            for item in provider_items
        ],
        "full_text": sanitize_text(record.get("text") or "") if role == "user" else "",
    }


def context_record_details_payload(record: dict[str, object], *, node_number: int) -> dict[str, object]:
    overview = context_record_overview(record, node_number=node_number)
    provider_items = normalize_provider_items(record.get("providerItems"))
    return {
        "node_number": node_number,
        "role": overview["role"],
        "token_estimate": overview["token_estimate"],
        "tool_token_estimate": overview["tool_token_estimate"],
        "tool_usage": overview["tool_usage"],
        "preview": overview["preview"],
        "item_count": len(provider_items),
        "text": sanitize_text(record.get("text") or ""),
        "attachments": sanitize_value(normalize_attachment_records(record.get("attachments"))),
        "blocks": [
            context_detail_block(block, block_number)
            for block_number, block in enumerate(normalize_message_blocks(record.get("blocks")), start=1)
        ],
        "provider_items": provider_items,
        "items": [
            provider_item_detail(item, item_number)
            for item_number, item in enumerate(provider_items, start=1)
        ],
    }


def build_context_workspace_snapshot(
    session: SessionState,
    *,
    selected_indexes: list[int] | None = None,
) -> str:
    safe_selected_indexes = normalize_selected_node_indexes(selected_indexes or [], len(session.transcript))
    selected_numbers = [index + 1 for index in safe_selected_indexes]
    lines = [
        "# 当前上下文快照",
        f"- 会话标题：{session.title}",
        f"- 会话类型：{session.scope}",
        f"- 当前节点数：{len(session.transcript)}",
        f"- 当前选中节点：{format_node_ranges(selected_numbers) or '未单独选中，默认面向全局'}",
        "- 这一轮里所有 Node # 都以这份快照为准。",
        "- user 节点直接给全文，assistant 节点默认只给概览。",
        "- 如果你需要 assistant 节点的完整协议层细节，再调用 get_context_node_details。",
        "",
        "## 节点概览",
    ]

    for node_number, record in enumerate(session.transcript, start=1):
        overview = context_record_overview(
            record,
            node_number=node_number,
            selected=(node_number - 1) in safe_selected_indexes,
        )
        marker = " | selected" if overview["selected"] else ""
        token_label = format_token_count(int(overview["token_estimate"] or 0))
        tool_token_estimate = int(overview.get("tool_token_estimate") or 0)
        tool_token_label = (
            f" | tool {format_token_count(tool_token_estimate)} tokens"
            if tool_token_estimate > 0
            else ""
        )
        if overview["role"] == "user":
            user_text = sanitize_text(overview["full_text"] or "").strip() or "[empty]"
            lines.append(f"- Node #{node_number} | user{marker} | {token_label} tokens")
            lines.append("  content:")
            for content_line in user_text.splitlines() or ["[empty]"]:
                lines.append(f"    {content_line}")
            continue

        lines.append(
            f"- Node #{node_number} | assistant{marker} | {token_label} tokens{tool_token_label} | {format_tool_usage(overview['tool_usage'])} | {int(overview['item_count'] or 0)} items"
        )
        lines.append(f"  preview: {sanitize_text(overview['preview'] or '') or '[empty]'}")

    return "\n".join(lines).strip()


def format_node_ranges(node_numbers: list[int]) -> str:
    if not node_numbers:
        return ""

    ordered = sorted(set(node_numbers))
    segments: list[str] = []
    range_start = ordered[0]
    previous = ordered[0]
    for current in ordered[1:]:
        if current == previous + 1:
            previous = current
            continue
        segments.append(f"{range_start}" if range_start == previous else f"{range_start}-{previous}")
        range_start = current
        previous = current
    segments.append(f"{range_start}" if range_start == previous else f"{range_start}-{previous}")
    return ", ".join(segments)


def letter_index(value: int) -> str:
    result = ""
    current = max(1, value)
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        result = f"{chr(65 + remainder)}{result}"
    return result


@dataclass(slots=True)
class ContextWorkbenchDraftNode:
    order: float
    label: str
    record: dict[str, object]
    active: bool
    source_node_number: int | None = None
    kind: str = "existing"
    status: str = "active"


class ContextWorkbenchDraft:
    def __init__(self, transcript: list[dict[str, object]], selected_indexes: list[int]) -> None:
        safe_selected = normalize_selected_node_indexes(selected_indexes, len(transcript))
        self.selected_node_numbers = [index + 1 for index in safe_selected]
        self.nodes = [
            ContextWorkbenchDraftNode(
                order=float(node_number),
                label=f"Node #{node_number}",
                record=sanitize_value(record),
                active=True,
                source_node_number=node_number,
            )
            for node_number, record in enumerate(transcript, start=1)
        ]
        self.operations: list[dict[str, object]] = []
        self._draft_counter = 0
        self._revision_summary = ""
        self._working_version = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.operations)

    def _record_operation(self, operation: dict[str, object]) -> None:
        self._working_version += 1
        operation["working_version"] = self._working_version
        self.operations.append(operation)
        self._revision_summary = ""

    def set_revision_summary(self, summary: str) -> dict[str, object]:
        if not self.operations:
            raise ValueError("no working snapshot edits exist yet")

        safe_summary = re.sub(r"\s+", " ", sanitize_text(summary)).strip()
        if not safe_summary:
            raise ValueError("summary is required")
        if len(safe_summary) > 220:
            safe_summary = f"{safe_summary[:219].rstrip()}…"

        self._revision_summary = safe_summary
        return {
            "payload_kind": "revision_summary",
            "saved": True,
            "summary": safe_summary,
            "change_count": len(self.operations),
            "working_version": self._working_version,
        }

    def _fallback_revision_summary(self) -> str:
        if not self.operations:
            return "这次更新了当前上下文。"
        return fallback_context_revision_summary("Context update", self.operations)

    def revision_summary(self) -> str:
        return self._revision_summary or self._fallback_revision_summary()

    def active_nodes(self) -> list[ContextWorkbenchDraftNode]:
        return [node for node in sorted(self.nodes, key=lambda item: item.order) if node.active]

    def max_node_number(self) -> int:
        return max((node.source_node_number or 0) for node in self.nodes) if self.nodes else 0

    def _nodes_by_number(self, node_numbers: list[int], *, include_inactive: bool = False) -> list[ContextWorkbenchDraftNode]:
        targets: list[ContextWorkbenchDraftNode] = []
        for node_number in node_numbers:
            node = next(
                (
                    item
                    for item in self.nodes
                    if item.source_node_number == node_number and (include_inactive or item.active)
                ),
                None,
            )
            if node is not None:
                targets.append(node)
        return targets

    def resolve_target_nodes(
        self,
        arguments: dict[str, Any],
        *,
        allow_all_active: bool = False,
        include_inactive: bool = False,
    ) -> list[ContextWorkbenchDraftNode]:
        explicit_numbers = normalize_node_numbers(arguments.get("node_numbers"), self.max_node_number())
        if explicit_numbers:
            return self._nodes_by_number(explicit_numbers, include_inactive=include_inactive)

        legacy_indexes = normalize_selected_node_indexes(arguments.get("node_indexes"), self.max_node_number())
        if legacy_indexes:
            return self._nodes_by_number([index + 1 for index in legacy_indexes], include_inactive=include_inactive)

        if allow_all_active:
            return self.active_nodes()

        return []

    def _overview_for_node(self, node: ContextWorkbenchDraftNode) -> dict[str, object]:
        display_number = node.source_node_number or 1
        overview = context_record_overview(
            node.record,
            node_number=display_number,
            selected=(node.source_node_number or 0) in self.selected_node_numbers,
        )
        overview["payload_kind"] = "node_overview"
        overview["node_number"] = node.source_node_number
        overview["label"] = node.label
        overview["status"] = node.status
        overview["node_kind"] = node.kind
        overview["active"] = node.active
        return overview

    def current_overview_items(self) -> list[dict[str, object]]:
        return [self._overview_for_node(node) for node in self.active_nodes()]

    def compact_overview_for_node(self, node: ContextWorkbenchDraftNode) -> dict[str, object]:
        overview = self._overview_for_node(node)
        overview.pop("full_text", None)
        return overview

    def compact_overview_items(self, nodes: list[ContextWorkbenchDraftNode]) -> list[dict[str, object]]:
        return [self.compact_overview_for_node(node) for node in nodes]

    def final_snapshot_payload(self) -> dict[str, object]:
        active_nodes = self.active_nodes()
        inactive_nodes = [node for node in sorted(self.nodes, key=lambda item: item.order) if not node.active]
        compressed_replacements: dict[int, str] = {}
        for operation in self.operations:
            if sanitize_text(operation.get("operation_type") or "").strip() != "compress_nodes":
                continue
            created_label = sanitize_text(operation.get("created_label") or "").strip()
            if not created_label:
                continue
            for node_number in unique_int_list(operation.get("compressed_node_numbers") or operation.get("target_node_numbers")):
                compressed_replacements[node_number] = created_label

        active_overviews = self.compact_overview_items(active_nodes)
        inactive_overviews: list[dict[str, object]] = []
        for node in inactive_nodes:
            item = {
                "node_number": node.source_node_number,
                "label": node.label,
                "status": node.status,
                "node_kind": node.kind,
                "active": node.active,
            }
            if node.status == "compressed" and node.source_node_number in compressed_replacements:
                item["replaced_by"] = compressed_replacements[node.source_node_number]
            inactive_overviews.append(item)

        return {
            "payload_kind": "final_working_snapshot",
            "working_version": self._working_version,
            "active_node_count": len(active_nodes),
            "inactive_node_count": len(inactive_nodes),
            "total_token_estimate": sum(int(item.get("token_estimate") or 0) for item in active_overviews),
            "tool_token_estimate": sum(int(item.get("tool_token_estimate") or 0) for item in active_overviews),
            "selected_node_numbers": list(self.selected_node_numbers),
            "active_nodes": active_overviews,
            "inactive_nodes": inactive_overviews,
            "operations": sanitize_value(self.operations),
        }

    def overview_items(self, nodes: list[ContextWorkbenchDraftNode]) -> list[dict[str, object]]:
        return [self._overview_for_node(node) for node in nodes]

    def node_details(self, nodes: list[ContextWorkbenchDraftNode]) -> list[dict[str, object]]:
        details: list[dict[str, object]] = []
        for node in nodes:
            detail = context_record_details_payload(node.record, node_number=node.source_node_number or 1)
            detail["payload_kind"] = "node_detail"
            detail["node_number"] = node.source_node_number
            detail["label"] = node.label
            detail["status"] = node.status
            detail["active"] = node.active
            detail["node_kind"] = node.kind
            details.append(detail)
        return details

    def mutation_node_details(self, nodes: list[ContextWorkbenchDraftNode]) -> list[dict[str, object]]:
        details: list[dict[str, object]] = []
        for node in nodes:
            provider_items = self._provider_items_for_node(node)
            overview = self._overview_for_node(node)
            details.append(
                {
                    "payload_kind": "node_mutation_detail",
                    "node_number": node.source_node_number,
                    "label": node.label,
                    "status": node.status,
                    "active": node.active,
                    "node_kind": node.kind,
                    "overview": overview,
                    "item_count": len(provider_items),
                    "full_detail_note": (
                        "Mutation results intentionally omit full provider_items and per-item detail to avoid repeating large node content. "
                        "For simple delete/replace/compress steps, do not re-open node details just to verify; use the mutation delta. "
                        "Only call get_context_node_details again when the next edit requires exact updated provider_items from the current working snapshot."
                    ),
                }
            )
        return details

    def _next_draft_label(self) -> str:
        self._draft_counter += 1
        return f"Draft Node {letter_index(self._draft_counter)}"

    def _set_node_record(self, node: ContextWorkbenchDraftNode, record: dict[str, object], *, status: str = "updated") -> None:
        normalized_record = normalize_transcript([record])
        if not normalized_record:
            raise ValueError("record could not be normalized after mutation")
        node.record = normalized_record[0]
        if node.kind == "existing":
            node.status = status

    def _provider_items_for_node(self, node: ContextWorkbenchDraftNode) -> list[dict[str, Any]]:
        return normalize_provider_items(node.record.get("providerItems"))

    def _resolve_item_detail(self, node: ContextWorkbenchDraftNode, item_number: int) -> dict[str, object]:
        items = self.node_details([node])[0].get("items")
        if not isinstance(items, list):
            raise ValueError("node detail items are unavailable")
        if item_number < 1 or item_number > len(items):
            raise ValueError(f"item #{item_number} does not exist in {node.label}")
        item = items[item_number - 1]
        if not isinstance(item, dict):
            raise ValueError(f"item #{item_number} could not be resolved in {node.label}")
        return item

    def _build_mutation_result(
        self,
        *,
        summary: str,
        change_type: str,
        changed_nodes: list[int],
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        changed_node_details = self.mutation_node_details(
            self._nodes_by_number(changed_nodes, include_inactive=True)
        )
        active_nodes = self.active_nodes()
        payload: dict[str, object] = {
            "payload_kind": "mutation_delta",
            "summary": summary,
            "change_type": normalize_change_type(change_type),
            "working_version": self._working_version,
            "changed_nodes": unique_int_list(changed_nodes),
            "active_node_count": len(active_nodes),
            "inactive_node_count": len([node for node in self.nodes if not node.active]),
            "changed_node_details": changed_node_details,
        }
        if extra:
            payload.update(sanitize_value(extra))
        return payload

    def delete_nodes(self, nodes: list[ContextWorkbenchDraftNode], *, reason: str) -> dict[str, object]:
        active_nodes = [node for node in nodes if node.active]
        if not active_nodes:
            raise ValueError("No active nodes were resolved for deletion.")

        deleted_numbers = [
            node.source_node_number
            for node in active_nodes
            if node.source_node_number is not None
        ]
        for node in active_nodes:
            node.active = False
            node.status = "deleted"

        summary = f"Delete nodes #{format_node_ranges(deleted_numbers)}"
        self._record_operation(
            {
                "operation_type": "delete_nodes",
                "change_type": "delete",
                "label": summary,
                "summary": summary,
                "changed_nodes": deleted_numbers,
                "target_node_numbers": deleted_numbers,
                "reason": sanitize_text(reason),
            }
        )
        return self._build_mutation_result(
            summary=summary,
            change_type="delete",
            changed_nodes=deleted_numbers,
            extra={
                "deleted_node_numbers": deleted_numbers,
            },
        )

    def compress_nodes(
        self,
        nodes: list[ContextWorkbenchDraftNode],
        *,
        summary_markdown: str,
        style: str,
        title: str,
    ) -> dict[str, object]:
        active_nodes = [node for node in nodes if node.active]
        if not active_nodes:
            raise ValueError("No active nodes were resolved for compression.")

        safe_summary = sanitize_text(summary_markdown).strip()
        if not safe_summary:
            raise ValueError("summary_markdown is required")

        target_numbers = [
            node.source_node_number
            for node in active_nodes
            if node.source_node_number is not None
        ]
        for node in active_nodes:
            node.active = False
            node.status = "compressed"

        label = self._next_draft_label()
        heading = sanitize_text(title).strip()
        summary_text = safe_summary if not heading else f"### {heading}\n\n{safe_summary}"
        created_node = ContextWorkbenchDraftNode(
            order=min(node.order for node in active_nodes) + 0.01,
            label=label,
            record={
                "role": "assistant",
                "text": summary_text,
                "attachments": [],
                "toolEvents": [],
                "blocks": [{"kind": "text", "text": summary_text}],
            },
            active=True,
            source_node_number=None,
            kind="draft",
            status="created",
        )
        self.nodes.append(created_node)

        summary = f"Compress nodes #{format_node_ranges(target_numbers)}"
        self._record_operation(
            {
                "operation_type": "compress_nodes",
                "change_type": "compress",
                "label": summary,
                "summary": summary,
                "changed_nodes": target_numbers,
                "target_node_numbers": target_numbers,
                "style": sanitize_text(style).strip(),
                "created_label": label,
            }
        )
        return self._build_mutation_result(
            summary=summary,
            change_type="compress",
            changed_nodes=target_numbers,
            extra={
                "compressed_node_numbers": target_numbers,
                "created_label": label,
                "created_node": self.compact_overview_for_node(created_node),
            },
        )

    def delete_item(self, node: ContextWorkbenchDraftNode, *, item_number: int, reason: str) -> dict[str, object]:
        provider_items = self._provider_items_for_node(node)
        removed_item = self._resolve_item_detail(node, item_number)
        del provider_items[item_number - 1]
        self._set_node_record(node, compile_record_from_provider_items(node.record, provider_items))

        changed_nodes = [node.source_node_number] if node.source_node_number is not None else []
        summary = f"Delete {node.label} item #{item_number}"
        self._record_operation(
            {
                "operation_type": "delete_item",
                "change_type": "delete",
                "label": summary,
                "summary": summary,
                "changed_nodes": changed_nodes,
                "target_node_numbers": changed_nodes,
                "target_items": [
                    {
                        "node_number": node.source_node_number,
                        "item_number": item_number,
                        "item_type": sanitize_text(removed_item.get("item_type") or ""),
                    }
                ],
                "reason": sanitize_text(reason).strip(),
            }
        )
        return self._build_mutation_result(
            summary=summary,
            change_type="delete",
            changed_nodes=changed_nodes,
            extra={
                "deleted_items": [
                    {
                        "node_number": node.source_node_number,
                        "item_number": item_number,
                        "item": removed_item,
                    }
                ],
            },
        )

    def replace_item(
        self,
        node: ContextWorkbenchDraftNode,
        *,
        item_number: int,
        replacement_item: dict[str, Any],
        reason: str,
        change_type: str = "replace",
    ) -> dict[str, object]:
        provider_items = self._provider_items_for_node(node)
        original_item = self._resolve_item_detail(node, item_number)
        normalized_replacement = normalize_provider_items([replacement_item])
        if len(normalized_replacement) != 1:
            raise ValueError("replacement_item must normalize into exactly one provider item")
        provider_items[item_number - 1] = normalized_replacement[0]
        self._set_node_record(node, compile_record_from_provider_items(node.record, provider_items))

        changed_nodes = [node.source_node_number] if node.source_node_number is not None else []
        summary_prefix = "Compress" if normalize_change_type(change_type) == "compress" else "Replace"
        summary = f"{summary_prefix} {node.label} item #{item_number}"
        self._record_operation(
            {
                "operation_type": "compress_item"
                if normalize_change_type(change_type) == "compress"
                else "replace_item",
                "change_type": normalize_change_type(change_type),
                "label": summary,
                "summary": summary,
                "changed_nodes": changed_nodes,
                "target_node_numbers": changed_nodes,
                "target_items": [
                    {
                        "node_number": node.source_node_number,
                        "item_number": item_number,
                        "item_type": sanitize_text(original_item.get("item_type") or ""),
                    }
                ],
                "replacement_item": sanitize_value(normalized_replacement[0]),
                "reason": sanitize_text(reason).strip(),
            }
        )
        return self._build_mutation_result(
            summary=summary,
            change_type=change_type,
            changed_nodes=changed_nodes,
            extra={
                "replaced_items": [
                    {
                        "node_number": node.source_node_number,
                        "item_number": item_number,
                        "before": provider_item_detail(original_item, item_number),
                        "after": provider_item_detail(normalized_replacement[0], item_number),
                    }
                ],
            },
        )

    def compress_item(
        self,
        node: ContextWorkbenchDraftNode,
        *,
        item_number: int,
        compressed_content: str,
        style: str,
    ) -> dict[str, object]:
        provider_items = self._provider_items_for_node(node)
        if item_number < 1 or item_number > len(provider_items):
            raise ValueError(f"item #{item_number} does not exist in {node.label}")

        original_item = provider_items[item_number - 1]
        item_type = sanitize_text(original_item.get("type") or "").strip()
        safe_content = sanitize_text(compressed_content).strip()
        if not safe_content:
            raise ValueError("compressed_content is required")

        replacement_item = sanitize_value(original_item)
        if item_type == "message":
            replacement_item["content"] = replace_provider_message_text(original_item.get("content"), safe_content)
        elif item_type == "function_call":
            replacement_item["arguments"] = safe_content
        elif item_type == "function_call_output":
            replacement_item["output"] = safe_content
        else:
            raise ValueError(f"{node.label} item #{item_number} cannot be compressed")

        return self.replace_item(
            node,
            item_number=item_number,
            replacement_item=replacement_item,
            reason=sanitize_text(style).strip(),
            change_type="compress",
        )

    def committed_transcript(self) -> list[dict[str, object]]:
        return normalize_transcript([node.record for node in self.active_nodes()])

    def revision_label(self) -> str:
        if not self.operations:
            return "Context update"
        if len(self.operations) == 1:
            return sanitize_text(self.operations[0].get("summary") or self.operations[0].get("label") or "").strip() or "Context update"
        first_label = sanitize_text(
            self.operations[0].get("summary") or self.operations[0].get("label") or ""
        ).strip() or "Context update"
        return f"{first_label} + {len(self.operations) - 1} more"


class ContextWorkbenchToolRegistry:
    def __init__(self, draft: ContextWorkbenchDraft) -> None:
        self._returned_detail_node_numbers: set[int] = set()
        self.draft = draft
        self._tools = {
            definition.name: definition
            for definition in [
                self._build_node_detail_tool(),
                self._build_delete_item_tool(),
                self._build_replace_item_tool(),
                self._build_compress_item_tool(),
                self._build_compress_nodes_tool(),
                self._build_delete_nodes_tool(),
                self._build_confirm_working_snapshot_tool(),
                self._build_set_revision_summary_tool(),
            ]
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    @classmethod
    def tool_catalog(cls) -> list[dict[str, str]]:
        return [
            {
                "id": "get_context_node_details",
                "label": "Node Details",
                "description": "Expand one or more nodes into full blocks and provider items before editing them.",
                "status": "available",
            },
            {
                "id": "delete_context_item",
                "label": "Delete Item",
                "description": "Delete one item inside a single node from the current working snapshot.",
                "status": "available",
            },
            {
                "id": "replace_context_item",
                "label": "Replace Item",
                "description": "Replace one item inside a single node with a new provider item.",
                "status": "available",
            },
            {
                "id": "compress_context_item",
                "label": "Compress Item",
                "description": "Replace one item with a shorter version while keeping the same item type.",
                "status": "available",
            },
            {
                "id": "compress_context_nodes",
                "label": "Compress Nodes",
                "description": "Replace one or more nodes with a new summary node inside the current working snapshot.",
                "status": "available",
            },
            {
                "id": "delete_context_nodes",
                "label": "Delete Nodes",
                "description": "Delete one or more nodes from the current working snapshot.",
                "status": "available",
            },
            {
                "id": "confirm_working_snapshot",
                "label": "Confirm Working Snapshot",
                "description": "Confirm the final overview of every active node after all intended edits are complete.",
                "status": "available",
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        tool = self._tools.get(name)
        if tool is None:
            return ToolExecution(
                output_text=json.dumps({"error": f"unknown workbench tool: {name}"}, ensure_ascii=False),
                display_title=name,
                display_detail="unknown context workbench tool",
                display_result="The requested context workbench tool does not exist.",
                status="error",
            )

        try:
            return tool.handler(arguments)
        except Exception as exc:  # noqa: BLE001
            return ToolExecution(
                output_text=json.dumps({"error": str(exc), "tool": name}, ensure_ascii=False),
                display_title=tool.label,
                display_detail="context workbench tool failed",
                display_result=sanitize_text(str(exc) or "The context workbench tool failed."),
                status="error",
            )

    def _target_resolution_execution(
        self,
        *,
        action_name: str,
        message: str,
        candidates: list[dict[str, object]] | None = None,
        requires_single_node: bool = False,
        should_expand_details: bool = False,
    ) -> ToolExecution:
        payload = {
            "payload_kind": "target_resolution",
            "resolved": False,
            "action": action_name,
            "message": message,
            "requires_single_node": requires_single_node,
            "should_expand_details": should_expand_details,
            "selected_node_numbers": list(self.draft.selected_node_numbers),
            "candidates": sanitize_value(candidates or []),
        }
        return ToolExecution(
            output_text=json.dumps(payload, ensure_ascii=False),
            display_title="Target Resolution",
            display_detail=action_name,
            display_result=message,
            status="needs_input",
        )

    def _item_resolution_execution(
        self,
        *,
        node: ContextWorkbenchDraftNode,
        item_number: int,
        message: str,
    ) -> ToolExecution:
        payload = {
            "payload_kind": "item_resolution",
            "resolved": False,
            "message": message,
            "requested_item_number": item_number,
            "node_detail": self.draft.node_details([node])[0],
        }
        return ToolExecution(
            output_text=json.dumps(payload, ensure_ascii=False),
            display_title="Item Resolution",
            display_detail=node.label,
            display_result=message,
            status="needs_input",
        )

    def _mark_detail_nodes_returned(self, nodes: list[ContextWorkbenchDraftNode]) -> None:
        for node in nodes:
            if node.source_node_number is not None:
                self._returned_detail_node_numbers.add(node.source_node_number)

    def _filter_new_detail_nodes(
        self,
        nodes: list[ContextWorkbenchDraftNode],
    ) -> tuple[list[ContextWorkbenchDraftNode], list[int]]:
        fresh_nodes: list[ContextWorkbenchDraftNode] = []
        cached_numbers: list[int] = []
        for node in nodes:
            if node.source_node_number is None:
                fresh_nodes.append(node)
                continue
            if node.source_node_number in self._returned_detail_node_numbers:
                cached_numbers.append(node.source_node_number)
                continue
            fresh_nodes.append(node)
        return fresh_nodes, cached_numbers

    def _invalidate_detail_cache(self) -> None:
        self._returned_detail_node_numbers.clear()

    def _build_node_detail_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(arguments: dict[str, Any]) -> ToolExecution:
            nodes = self.draft.resolve_target_nodes(arguments)
            if not nodes:
                return self._target_resolution_execution(
                    action_name="get_context_node_details",
                    message="get_context_node_details requires explicit node_numbers from the current snapshot.",
                    should_expand_details=True,
                )

            fresh_nodes, cached_node_numbers = self._filter_new_detail_nodes(nodes)
            details = self.draft.node_details(fresh_nodes)
            self._mark_detail_nodes_returned(fresh_nodes)
            labels = ", ".join(
                sanitize_text(item.get("label") or "").strip()
                for item in details
                if sanitize_text(item.get("label") or "").strip()
            )
            cached_label = format_node_ranges(cached_node_numbers)
            display_result_parts: list[str] = []
            if labels:
                display_result_parts.append(f"Returned details for {labels}.")
            if cached_label:
                display_result_parts.append(
                    f"Skipped duplicate details for Node #{cached_label}; use the previous result from this turn."
                )
            return ToolExecution(
                output_text=json.dumps(
                    {
                        "payload_kind": "node_detail_list",
                        "selected_node_numbers": list(self.draft.selected_node_numbers),
                        "items": details,
                        "cached_node_numbers": cached_node_numbers,
                        "cached_message": (
                            f"Node #{cached_label} details were already returned earlier in this same workbench turn. "
                            "Use the previous function_call_output for those nodes."
                            if cached_node_numbers
                            else ""
                        ),
                    },
                    ensure_ascii=False,
                ),
                display_title="Node Details",
                display_detail=labels or "node details",
                display_result=" ".join(display_result_parts)
                or "The requested node details were already returned earlier in this turn.",
            )

        return ContextWorkbenchToolDefinition(
            name="get_context_node_details",
            label="Node Details",
            description="Expand one or more nodes into full blocks and provider items before editing them.",
            parameters={
                "type": "object",
                "properties": {
                    "node_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Required 1-based Node # values from the current snapshot.",
                    },
                },
                "required": ["node_numbers"],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )

    def _build_delete_item_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(arguments: dict[str, Any]) -> ToolExecution:
            item_number = int(arguments.get("item_number") or 0)
            nodes = self.draft.resolve_target_nodes(arguments)
            if not nodes:
                return self._target_resolution_execution(
                    action_name="delete_context_item",
                    message="delete_context_item requires exactly one explicit node_numbers value from the current snapshot.",
                    requires_single_node=True,
                    should_expand_details=True,
                )
            if len(nodes) != 1:
                return self._target_resolution_execution(
                    action_name="delete_context_item",
                    message="delete_context_item needs exactly one target node. Narrow it to a single Node # first.",
                    candidates=self.draft.overview_items(nodes),
                    requires_single_node=True,
                    should_expand_details=True,
                )

            node = nodes[0]
            try:
                self.draft._resolve_item_detail(node, item_number)
            except ValueError as exc:
                return self._item_resolution_execution(node=node, item_number=item_number, message=str(exc))

            result = self.draft.delete_item(
                node,
                item_number=item_number,
                reason=sanitize_text(arguments.get("reason") or "").strip(),
            )
            self._invalidate_detail_cache()
            return ToolExecution(
                output_text=json.dumps(result, ensure_ascii=False),
                display_title="Delete Item",
                display_detail=result["summary"],
                display_result=result["summary"],
            )

        return ContextWorkbenchToolDefinition(
            name="delete_context_item",
            label="Delete Item",
            description="Delete one item inside a single node from the current working snapshot.",
            parameters={
                "type": "object",
                "properties": {
                    "node_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Required single 1-based Node # value from the current snapshot.",
                    },
                    "item_number": {
                        "type": "integer",
                        "description": "Required item # inside the resolved node.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for deleting this item.",
                    },
                },
                "required": ["node_numbers", "item_number"],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )

    def _replacement_item_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["message", "function_call", "function_call_output"],
                },
                "role": {
                    "type": "string",
                    "enum": ["system", "developer", "user", "assistant"],
                },
                "content": {
                    "type": ["string", "array"],
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "call_id": {
                    "type": "string",
                },
                "name": {
                    "type": "string",
                },
                "arguments": {
                    "type": "string",
                },
                "output": {
                    "type": "string",
                },
            },
            "required": ["type"],
            "additionalProperties": False,
        }

    def _build_replace_item_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(arguments: dict[str, Any]) -> ToolExecution:
            item_number = int(arguments.get("item_number") or 0)
            nodes = self.draft.resolve_target_nodes(arguments)
            if not nodes:
                return self._target_resolution_execution(
                    action_name="replace_context_item",
                    message="replace_context_item requires exactly one explicit node_numbers value from the current snapshot.",
                    requires_single_node=True,
                    should_expand_details=True,
                )
            if len(nodes) != 1:
                return self._target_resolution_execution(
                    action_name="replace_context_item",
                    message="replace_context_item needs exactly one target node. Narrow it to a single Node # first.",
                    candidates=self.draft.overview_items(nodes),
                    requires_single_node=True,
                    should_expand_details=True,
                )

            node = nodes[0]
            try:
                self.draft._resolve_item_detail(node, item_number)
            except ValueError as exc:
                return self._item_resolution_execution(node=node, item_number=item_number, message=str(exc))

            replacement_item = arguments.get("replacement_item")
            if not isinstance(replacement_item, dict):
                return self._item_resolution_execution(
                    node=node,
                    item_number=item_number,
                    message="replacement_item must be an object that matches one editable provider item.",
                )

            result = self.draft.replace_item(
                node,
                item_number=item_number,
                replacement_item=replacement_item,
                reason=sanitize_text(arguments.get("reason") or "").strip(),
            )
            self._invalidate_detail_cache()
            return ToolExecution(
                output_text=json.dumps(result, ensure_ascii=False),
                display_title="Replace Item",
                display_detail=result["summary"],
                display_result=result["summary"],
            )

        return ContextWorkbenchToolDefinition(
            name="replace_context_item",
            label="Replace Item",
            description="Replace one item inside a single node with a new provider item.",
            parameters={
                "type": "object",
                "properties": {
                    "node_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Required single 1-based Node # value from the current snapshot.",
                    },
                    "item_number": {
                        "type": "integer",
                        "description": "Required item # inside the resolved node.",
                    },
                    "replacement_item": self._replacement_item_schema(),
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for replacing this item.",
                    },
                },
                "required": ["node_numbers", "item_number", "replacement_item"],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )

    def _build_compress_item_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(arguments: dict[str, Any]) -> ToolExecution:
            item_number = int(arguments.get("item_number") or 0)
            nodes = self.draft.resolve_target_nodes(arguments)
            if not nodes:
                return self._target_resolution_execution(
                    action_name="compress_context_item",
                    message="compress_context_item requires exactly one explicit node_numbers value from the current snapshot.",
                    requires_single_node=True,
                    should_expand_details=True,
                )
            if len(nodes) != 1:
                return self._target_resolution_execution(
                    action_name="compress_context_item",
                    message="compress_context_item needs exactly one target node. Narrow it to a single Node # first.",
                    candidates=self.draft.overview_items(nodes),
                    requires_single_node=True,
                    should_expand_details=True,
                )

            node = nodes[0]
            try:
                self.draft._resolve_item_detail(node, item_number)
            except ValueError as exc:
                return self._item_resolution_execution(node=node, item_number=item_number, message=str(exc))

            result = self.draft.compress_item(
                node,
                item_number=item_number,
                compressed_content=sanitize_text(arguments.get("compressed_content") or ""),
                style=sanitize_text(arguments.get("style") or "").strip(),
            )
            self._invalidate_detail_cache()
            return ToolExecution(
                output_text=json.dumps(result, ensure_ascii=False),
                display_title="Compress Item",
                display_detail=result["summary"],
                display_result=result["summary"],
            )

        return ContextWorkbenchToolDefinition(
            name="compress_context_item",
            label="Compress Item",
            description="Replace one item with a shorter version while keeping the same item type.",
            parameters={
                "type": "object",
                "properties": {
                    "node_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Required single 1-based Node # value from the current snapshot.",
                    },
                    "item_number": {
                        "type": "integer",
                        "description": "Required item # inside the resolved node.",
                    },
                    "compressed_content": {
                        "type": "string",
                        "description": "The shorter replacement content for this item.",
                    },
                    "style": {
                        "type": "string",
                        "description": "Optional note about the compression style.",
                    },
                },
                "required": ["node_numbers", "item_number", "compressed_content"],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )

    def _build_compress_nodes_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(arguments: dict[str, Any]) -> ToolExecution:
            nodes = self.draft.resolve_target_nodes(arguments)
            if not nodes:
                return self._target_resolution_execution(
                    action_name="compress_context_nodes",
                    message="compress_context_nodes requires explicit node_numbers from the current snapshot.",
                )

            result = self.draft.compress_nodes(
                nodes,
                summary_markdown=sanitize_text(arguments.get("summary_markdown") or ""),
                style=sanitize_text(arguments.get("style") or "").strip() or "tight summary",
                title=sanitize_text(arguments.get("title") or "").strip(),
            )
            self._invalidate_detail_cache()
            return ToolExecution(
                output_text=json.dumps(result, ensure_ascii=False),
                display_title="Compress Nodes",
                display_detail=result["summary"],
                display_result=result["summary"],
            )

        return ContextWorkbenchToolDefinition(
            name="compress_context_nodes",
            label="Compress Nodes",
            description="Replace one or more nodes with a new summary node inside the current working snapshot.",
            parameters={
                "type": "object",
                "properties": {
                    "node_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Required 1-based Node # values from the current snapshot.",
                    },
                    "summary_markdown": {
                        "type": "string",
                        "description": "Markdown content that should become the new summary node.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional heading for the created summary node.",
                    },
                    "style": {
                        "type": "string",
                        "description": "Short note about the compression style.",
                    },
                },
                "required": ["node_numbers", "summary_markdown"],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )

    def _build_delete_nodes_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(arguments: dict[str, Any]) -> ToolExecution:
            nodes = self.draft.resolve_target_nodes(arguments)
            if not nodes:
                return self._target_resolution_execution(
                    action_name="delete_context_nodes",
                    message="delete_context_nodes requires explicit node_numbers from the current snapshot.",
                )

            result = self.draft.delete_nodes(
                nodes,
                reason=sanitize_text(arguments.get("reason") or "").strip(),
            )
            self._invalidate_detail_cache()
            return ToolExecution(
                output_text=json.dumps(result, ensure_ascii=False),
                display_title="Delete Nodes",
                display_detail=result["summary"],
                display_result=result["summary"],
            )

        return ContextWorkbenchToolDefinition(
            name="delete_context_nodes",
            label="Delete Nodes",
            description="Delete one or more nodes from the current working snapshot.",
            parameters={
                "type": "object",
                "properties": {
                    "node_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Required 1-based Node # values from the current snapshot.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional reason for deleting these nodes.",
                    },
                },
                "required": ["node_numbers"],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )

    def _build_confirm_working_snapshot_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(_arguments: dict[str, Any]) -> ToolExecution:
            result = self.draft.final_snapshot_payload()
            active_count = int(result.get("active_node_count") or 0)
            inactive_count = int(result.get("inactive_node_count") or 0)
            total_tokens = int(result.get("total_token_estimate") or 0)
            return ToolExecution(
                output_text=json.dumps(result, ensure_ascii=False),
                display_title="Confirm Working Snapshot",
                display_detail=f"{active_count} active nodes, {inactive_count} inactive nodes",
                display_result=(
                    f"Confirmed final working snapshot: {active_count} active nodes, "
                    f"{inactive_count} inactive nodes, about {format_token_count(total_tokens)} tokens."
                ),
            )

        return ContextWorkbenchToolDefinition(
            name="confirm_working_snapshot",
            label="Confirm Working Snapshot",
            description="Confirm the final overview of every active node after all intended edits are complete. Use once near the end of a turn, not after every edit.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )

    def _build_set_revision_summary_tool(self) -> ContextWorkbenchToolDefinition:
        def handler(arguments: dict[str, Any]) -> ToolExecution:
            try:
                result = self.draft.set_revision_summary(
                    sanitize_text(arguments.get("summary") or ""),
                )
            except ValueError as exc:
                return ToolExecution(
                    output_text=json.dumps(
                        {
                            "payload_kind": "revision_summary",
                            "saved": False,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    display_title="Revision Summary",
                    display_detail="summary not saved",
                    display_result=str(exc),
                    status="needs_input",
                )

            return ToolExecution(
                output_text=json.dumps(result, ensure_ascii=False),
                display_title="Revision Summary",
                display_detail="saved",
                display_result=result["summary"],
            )

        return ContextWorkbenchToolDefinition(
            name="set_context_revision_summary",
            label="Revision Summary",
            description="After finishing working-snapshot edits, save one short summary (matching user language) that explains what this commit changed. Describe the content changed (e.g. 'compressed tool outputs'), not the node numbers. This text will be shown in the restore history.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One short summary (matching user language) of the content that changed in the context snapshot.",
                    },
                },
                "required": ["summary"],
                "additionalProperties": False,
            },
            status="available",
            handler=handler,
        )


def normalize_context_chat_history(raw_history: Any) -> list[dict[str, str]]:
    if not isinstance(raw_history, list):
        return []

    history: list[dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        role = sanitize_text(item.get("role") or "").strip()
        if role not in {"user", "assistant"}:
            continue
        content = sanitize_text(item.get("content") or "").strip()
        if not content:
            continue
        history.append(
            {
                "role": role,
                "content": content,
            }
        )
    return history


def prepare_context_chat_history_for_model(raw_history: Any, *, limit: int = 12) -> list[dict[str, str]]:
    history = normalize_context_chat_history(raw_history)
    filtered: list[dict[str, str]] = []

    for item in history:
        if item["role"] == "assistant":
            content = sanitize_text(item["content"])
            if "我已经读完当前上下文了，但这次没能稳定产出文字答复" in content:
                continue
        filtered.append(item)

    if limit > 0:
        return filtered[-limit:]
    return filtered
