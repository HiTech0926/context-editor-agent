from __future__ import annotations

import pytest

from agent_runtime.adapters.base import ProviderRequestContext
from agent_runtime.adapters.claude_adapter import ClaudeAdapter
from simple_agent.agent import SimpleAgent


def test_build_request_converts_input_image_data_url_to_claude_image_block() -> None:
    adapter = ClaudeAdapter(client=object())

    request = adapter.build_request(
        ProviderRequestContext(
            model="claude-sonnet-4-5",
            current_turn=(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this."},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,aGVsbG8=",
                            "detail": "auto",
                        },
                    ],
                },
            ),
        )
    )

    content = request["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this."}
    assert content[1] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "aGVsbG8=",
        },
    }


def test_build_request_converts_image_url_part_to_claude_url_image_block() -> None:
    adapter = ClaudeAdapter(client=object())

    request = adapter.build_request(
        ProviderRequestContext(
            model="claude-opus-4-7",
            current_turn=(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.webp",
                            },
                        },
                    ],
                },
            ),
        )
    )

    assert request["messages"][0]["content"] == [
        {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.webp",
            },
        }
    ]


def test_claude_agent_allows_image_content_parts_before_adapter_translation() -> None:
    agent = SimpleAgent.__new__(SimpleAgent)
    agent.provider_type = "claude"

    agent._assert_supported_content_parts(
        [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Look at this."},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,abc"},
                ],
            }
        ]
    )


def test_claude_rejects_unsupported_image_media_types() -> None:
    adapter = ClaudeAdapter(client=object())

    with pytest.raises(ValueError, match="Claude image input supports"):
        adapter.build_request(
            ProviderRequestContext(
                model="claude-sonnet-4-5",
                current_turn=(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/svg+xml;base64,PHN2Zy8+",
                            },
                        ],
                    },
                ),
            )
        )
