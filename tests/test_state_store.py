import json
import sqlite3

from web_server_modules.state_store import SQLiteStateStore


def sample_state() -> dict[str, object]:
    return {
        "projects": [
            {
                "id": "project-1",
                "title": "Demo",
                "session_ids": ["session-project"],
                "archived_session_ids": ["session-archived"],
                "root_path": "C:\\work\\demo",
            }
        ],
        "chat_session_ids": ["session-chat"],
        "sessions": {
            "session-chat": {
                "title": "Chat",
                "scope": "chat",
                "project_id": None,
                "transcript": [{"role": "user", "text": "hello"}],
                "context_workbench_history": [],
                "context_revisions": [{"id": "rev-1", "label": "Initial"}],
                "pending_context_restore": None,
            },
            "session-project": {
                "title": "Project Chat",
                "scope": "project",
                "project_id": "project-1",
                "transcript": [{"role": "assistant", "text": "done"}],
                "context_workbench_history": [{"role": "user", "content": "trim"}],
                "context_revisions": [],
                "pending_context_restore": {"target_revision_id": "rev-1"},
            },
            "session-archived": {
                "title": "Archived",
                "scope": "project",
                "project_id": "project-1",
                "transcript": [],
                "context_workbench_history": [],
                "context_revisions": [],
                "pending_context_restore": None,
            },
        },
    }


def test_state_store_migrates_legacy_json_to_sqlite(tmp_path):
    legacy_json = tmp_path / "hash_web_state.json"
    db_file = tmp_path / "hash_web_state.sqlite3"
    legacy_json.write_text(json.dumps(sample_state(), ensure_ascii=False), encoding="utf-8")

    store = SQLiteStateStore(db_file, legacy_json_file=legacy_json)

    assert store.load_state() == sample_state()
    assert db_file.exists()

    with sqlite3.connect(db_file) as connection:
        project_count = connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        session_count = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    assert project_count == 1
    assert session_count == 3


def test_state_store_round_trips_state_through_sqlite(tmp_path):
    db_file = tmp_path / "hash_web_state.sqlite3"
    store = SQLiteStateStore(db_file)

    store.save_state(sample_state())

    reloaded_store = SQLiteStateStore(db_file)
    assert reloaded_store.load_state() == sample_state()
