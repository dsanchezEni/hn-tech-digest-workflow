import asyncio
import json
from datetime import timedelta

from mistralai.workflows.testing import create_test_worker
from worker.workflows import github_repo_catchup as mod

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)

FAKE_MERGED = {
    "total_count": 2,
    "items": [
        {
            "number": 12,
            "title": "Fix login redirect loop",
            "body": "The redirect loop happened when the session cookie was expired.",
            "html_url": "https://github.com/octo/hello/pull/12",
            "user": {"login": "alice"},
            "closed_at": "2026-11-20T10:00:00Z",
            "labels": [{"name": "bug"}],
        },
        {
            "number": 11,
            "title": "Add dark theme",
            "body": "",
            "html_url": "https://github.com/octo/hello/pull/11",
            "user": {"login": "bob"},
            "closed_at": "2026-11-18T09:00:00Z",
            "labels": [],
        },
    ],
}

FAKE_OPEN = {
    "total_count": 1,
    "items": [
        {
            "number": 15,
            "title": "Refactor auth module",
            "body": "Needs review from the security team.",
            "html_url": "https://github.com/octo/hello/pull/15",
            "user": {"login": "carol"},
            "closed_at": None,
            "labels": [{"name": "enhancement"}],
        }
    ],
}

FAKE_SUMMARIES = {
    "merged": {
        "12": "Corrige la boucle de redirection après expiration du cookie de session.",
        "11": "Ajoute un thème sombre à l'interface.",
    },
    "pending": {
        "15": "Refonte du module d'authentification, en attente de revue sécurité.",
    },
}


def test_parse_repo_ok() -> None:
    assert mod._parse_repo("octo/hello") == ("octo", "hello")
    assert mod._parse_repo(" octo/hello ") == ("octo", "hello")


def test_parse_repo_rejects_bad_format() -> None:
    for bad in ("octo", "octo/hello/extra", "", "/"):
        try:
            mod._parse_repo(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"_parse_repo devrait rejeter {bad!r}")


def test_validate_since() -> None:
    assert mod._validate_since("2026-11-01") == "2026-11-01"
    for bad in ("01/11/2026", "2026-11-1", "not-a-date", ""):
        try:
            mod._validate_since(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"_validate_since devrait rejeter {bad!r}")


def test_tool_payload_dict_with_content_blocks() -> None:
    inner = {"total_count": 1, "items": [{"number": 1}]}
    result = {"content": [{"type": "text", "text": json.dumps(inner)}]}
    assert mod._tool_payload(result) == inner


def test_tool_payload_plain_dict_and_list() -> None:
    assert mod._tool_payload({"items": []}) == {"items": []}
    assert mod._tool_payload([{"type": "text", "text": '{"a": 2}'}]) == {"a": 2}
    assert mod._tool_payload("raw text") == "raw text"


def test_pr_items_and_normalize() -> None:
    items = mod._pr_items({"content": [{"type": "text", "text": json.dumps(FAKE_MERGED)}]})
    assert len(items) == 2
    normalized = mod._normalize_pr(items[0])
    assert normalized["number"] == 12
    assert normalized["title"] == "Fix login redirect loop"
    assert normalized["url"] == "https://github.com/octo/hello/pull/12"
    assert normalized["author"] == "alice"
    assert normalized["labels"] == ["bug"]
    assert mod._pr_items({"content": []}) == []
    assert mod._pr_items("oops") == []


def test_normalize_pr_truncates_body() -> None:
    item = {"number": 1, "title": "t", "body": "x" * (mod.MAX_PR_BODY_CHARS + 100)}
    assert len(mod._normalize_pr(item)["body"]) == mod.MAX_PR_BODY_CHARS


def test_build_summaries_falls_back_to_title() -> None:
    prs = [{"number": 7, "title": "Some title", "url": "https://x/7", "body": ""}]
    out = mod._build_summaries(prs, {"7": "résumé"})
    assert out[0].one_line == "résumé"
    out = mod._build_summaries(prs, {})
    assert out[0].one_line == "Some title"


def test_render_message_both_sections() -> None:
    report = mod.CatchupReport(
        merged=[
            mod.PrSummary(
                number=12,
                title="Fix login redirect loop",
                url="https://github.com/octo/hello/pull/12",
                one_line="Corrige la boucle de redirection.",
            )
        ],
        pending=[],
    )
    message = mod._render_message(report, "octo/hello", "2026-11-01", "2026-11-27 12:00 UTC")
    assert "octo/hello" in message
    assert "2026-11-01" in message
    assert "#12 Fix login redirect loop" in message
    assert "Corrige la boucle de redirection." in message
    assert "https://github.com/octo/hello/pull/12" in message
    assert "En attente de revue (0)" in message
    assert "Aucun pull request ouvert" in message


async def _fake_github_search(client, owner, repo, query):
    assert (owner, repo) == ("octo", "hello")
    if "is:merged" in query:
        assert "merged:>=2026-11-01" in query
        return [mod._normalize_pr(i) for i in FAKE_MERGED["items"]]
    assert "is:open" in query and "-is:draft" in query
    return [mod._normalize_pr(i) for i in FAKE_OPEN["items"]]


async def _fake_chat_complete_json(system, user):
    assert "UNE SEULE ligne" in system
    assert "octo/hello" in user
    assert "Fix login redirect loop" in user
    return FAKE_SUMMARIES


async def _fake_teams_agent(message):
    assert "Rattrapage GitHub" in message
    return "chat=19:abc message=1789xyz"


async def test_github_repo_catchup_dry_run(monkeypatch, temporal_env) -> None:
    monkeypatch.setattr(mod, "_github_search_prs", _fake_github_search)
    monkeypatch.setattr(mod, "_chat_complete_json", _fake_chat_complete_json)

    async with create_test_worker(
        temporal_env,
        workflows=[mod.GithubRepoCatchupWorkflow],
        activities=[
            mod.fetch_merged_prs,
            mod.fetch_open_prs,
            mod.summarize_prs,
            mod.post_to_teams,
        ],
    ):
        handle = await temporal_env.client.start_workflow(
            "github-repo-catchup",
            {"repo": "octo/hello", "since": "2026-11-01", "dry_run": True},
            id="test-catchup-dry-run",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)

    assert result["repo"] == "octo/hello"
    assert result["since"] == "2026-11-01"
    assert result["merged_count"] == 2
    assert result["pending_count"] == 1
    assert result["teams_posted"] is False
    assert "non posté" in result["teams_confirmation"]
    assert "#12 Fix login redirect loop" in result["message"]
    assert "https://github.com/octo/hello/pull/12" in result["message"]
    assert "#15 Refactor auth module" in result["message"]
    assert "Corrige la boucle de redirection" in result["message"]


async def test_github_repo_catchup_posts_to_teams(monkeypatch, temporal_env) -> None:
    posted: list[str] = []

    async def fake_teams_agent(message):
        posted.append(message)
        return await _fake_teams_agent(message)

    monkeypatch.setattr(mod, "_github_search_prs", _fake_github_search)
    monkeypatch.setattr(mod, "_chat_complete_json", _fake_chat_complete_json)
    monkeypatch.setattr(mod, "_run_teams_agent", fake_teams_agent)

    async with create_test_worker(
        temporal_env,
        workflows=[mod.GithubRepoCatchupWorkflow],
        activities=[
            mod.fetch_merged_prs,
            mod.fetch_open_prs,
            mod.summarize_prs,
            mod.post_to_teams,
        ],
    ):
        handle = await temporal_env.client.start_workflow(
            "github-repo-catchup",
            {"repo": "octo/hello", "since": "2026-11-01"},
            id="test-catchup-post",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)

    assert result["teams_posted"] is True
    assert result["teams_confirmation"] == "chat=19:abc message=1789xyz"
    assert len(posted) == 1
    assert "En attente de revue (1)" in posted[0]


async def test_github_repo_catchup_rejects_bad_input(monkeypatch, temporal_env) -> None:
    monkeypatch.setattr(mod, "_github_search_prs", _fake_github_search)
    monkeypatch.setattr(mod, "_chat_complete_json", _fake_chat_complete_json)

    async with create_test_worker(
        temporal_env,
        workflows=[mod.GithubRepoCatchupWorkflow],
        activities=[
            mod.fetch_merged_prs,
            mod.fetch_open_prs,
            mod.summarize_prs,
            mod.post_to_teams,
        ],
    ):
        handle = await temporal_env.client.start_workflow(
            "github-repo-catchup",
            {"repo": "not-a-repo", "since": "2026-11-01"},
            id="test-catchup-bad-input",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        try:
            await asyncio.wait_for(handle.result(), timeout=15)
        except Exception:
            pass
        else:
            raise AssertionError("une entrée invalide devrait faire échouer l'exécution")
