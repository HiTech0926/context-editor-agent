from __future__ import annotations

import json

from dotenv import load_dotenv

from simple_agent.agent import SimpleAgent
from simple_agent.config import load_settings


def sanitize_terminal_text(value: str) -> str:
    cleaned: list[str] = []
    for char in value:
        codepoint = ord(char)
        if 0xD800 <= codepoint <= 0xDFFF:
            cleaned.append(f"\\u{codepoint:04x}")
            continue
        cleaned.append(char)
    return "".join(cleaned)


def main() -> None:
    load_dotenv()
    settings = load_settings()
    agent = SimpleAgent(settings)

    print("Simple Responses Agent")
    print(f"model: {settings.model}")
    print(f"project root: {settings.project_root}")
    print("commands: /tools /reset /quit")

    while True:
        try:
            user_message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_message:
            continue

        if user_message in {"/quit", "quit", "exit"}:
            print("bye")
            break

        if user_message == "/reset":
            agent.reset()
            print("session cleared")
            continue

        if user_message == "/tools":
            print(agent.describe_tools())
            continue

        try:
            answer, tool_events = agent.run_turn(user_message)
        except Exception as exc:
            print(f"\nagent> error: {exc}")
            continue

        for event in tool_events:
            args_text = json.dumps(event.arguments, ensure_ascii=False)
            print(sanitize_terminal_text(f"[tool] {event.name}({args_text})"))
            print(sanitize_terminal_text(f"       -> {event.output_preview}"))

        print(sanitize_terminal_text(f"\nagent> {answer}"))


if __name__ == "__main__":
    main()
