import asyncio
import base64
from datetime import timedelta
from email import message_from_bytes

import pytest
from mistralai.workflows.testing import create_test_worker
from worker.workflows import hn_tech_digest as mod

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)

FAKE_HITS = {
    "hits": [
        {
            "objectID": "1",
            "title": "New AI chip announced by a major vendor",
            "url": "https://example.com/chip",
            "author": "alice",
            "points": 500,
            "num_comments": 200,
        },
        {
            "objectID": "2",
            "title": "My personal blog post about cooking",
            "url": "https://blog.example.com/cooking",
            "author": "bob",
            "points": 5,
            "num_comments": 1,
        },
        {
            "objectID": "3",
            "title": "Ask HN: what is the best laptop?",
            "url": None,
            "author": "carol",
            "points": 50,
            "num_comments": 80,
        },
        {
            "objectID": "4",
            "title": "Major open source project releases version 2.0",
            "url": "https://example.com/oss",
            "author": "dave",
            "points": 300,
            "num_comments": 90,
        },
    ]
}

FAKE_DIGEST = {
    "subject": "Veille tech Hacker News",
    "email_body": (
        "Intro.\n\n### New AI chip announced by a major vendor\nRésumé.\n\n"
        "### Major open source project releases version 2.0\nRésumé."
    ),
    "podcast_script": (
        "Voici les principales actualités tech du jour sur Hacker News. "
        "Un fondeur annonce une nouvelle puce dédiée à l'IA. "
        "Un projet open source majeur passe en version 2.0. Bonne écoute."
    ),
    "stories": [
        {
            "title": "New AI chip announced by a major vendor",
            "url": "https://example.com/chip",
            "summary": "Résumé.",
            "points": 500,
            "num_comments": 200,
        },
        {
            "title": "Major open source project releases version 2.0",
            "url": "https://example.com/oss",
            "summary": "Résumé.",
            "points": 300,
            "num_comments": 90,
        },
    ],
}


async def _fake_http_get_json(url, params):
    assert params["tags"] == "front_page"
    return FAKE_HITS


async def _fake_chat_complete_json(system, user):
    assert "secteur technologique" in system
    assert "top_n = 10" in user
    return FAKE_DIGEST


async def _fake_tts_mp3(script, voice_id):
    assert voice_id == mod.DEFAULT_VOICE_ID
    assert "Hacker News" in script
    return "QUJD" * 64


def test_parse_json_object_plain() -> None:
    assert mod._parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_wrapped() -> None:
    assert mod._parse_json_object('Voici le JSON :\n{"a": 1}\nfin') == {"a": 1}


def test_build_raw_mime() -> None:
    raw = mod._build_raw_mime(
        "a@b.c",
        "Sujet",
        "Corps du mail",
        base64.b64encode(b"MP3DATA").decode(),
        "pod.mp3",
    )
    msg = message_from_bytes(base64.urlsafe_b64decode(raw))
    assert msg["To"] == "a@b.c"
    assert msg["Subject"] == "Sujet"
    texts = [p for p in msg.walk() if p.get_content_type() == "text/plain"]
    assert texts and "Corps du mail" in texts[0].get_payload(decode=True).decode()
    attachments = [p for p in msg.walk() if p.get_content_maintype() == "audio"]
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "pod.mp3"
    assert attachments[0].get_payload(decode=True) == b"MP3DATA"


async def test_gmail_access_token_requires_secrets(monkeypatch) -> None:
    for name in mod.GMAIL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="GMAIL_CLIENT_ID"):
        await mod._gmail_access_token()


async def test_hn_tech_digest_end_to_end(monkeypatch, temporal_env) -> None:
    sent: dict = {}

    async def fake_gmail_send(to, subject, body, audio_b64, filename):
        sent.update(
            to=to, subject=subject, body=body, audio_b64=audio_b64, filename=filename
        )
        return "msg-123"

    monkeypatch.setattr(mod, "_http_get_json", _fake_http_get_json)
    monkeypatch.setattr(mod, "_chat_complete_json", _fake_chat_complete_json)
    monkeypatch.setattr(mod, "_tts_mp3", _fake_tts_mp3)
    monkeypatch.setattr(mod, "_gmail_send_email", fake_gmail_send)

    async with create_test_worker(
        temporal_env,
        workflows=[mod.HnTechDigestWorkflow],
        activities=[
            mod.fetch_front_page,
            mod.build_digest,
            mod.synthesize_podcast,
            mod.send_digest_email,
        ],
    ):
        handle = await temporal_env.client.start_workflow(
            "hn-tech-digest",
            {},
            id="test-hn-tech-digest",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=15)

    assert result["message_id"] == "msg-123"
    assert result["subject"] == "Veille tech Hacker News"
    assert result["stories_count"] == 2
    assert result["audio_b64_chars"] == len("QUJD" * 64)
    assert sent["to"] == "dsanchez@campus-eni.fr"
    assert sent["filename"].startswith("hn-tech-digest-")
    assert sent["filename"].endswith(".mp3")
    assert sent["audio_b64"] == "QUJD" * 64
    assert "New AI chip announced by a major vendor" in sent["body"]
