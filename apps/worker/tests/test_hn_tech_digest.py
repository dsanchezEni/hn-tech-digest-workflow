import asyncio
import json
from datetime import UTC, datetime, timedelta

from mistralai.workflows.testing import create_test_worker
from worker.workflows import hn_tech_digest as mod

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)
NOW = datetime(2026, 11, 27, 12, 0, 0, tzinfo=UTC)

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

FAKE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>TechCrunch</title>
<item>
  <title>Startup raises big round for AI agents</title>
  <link>https://techcrunch.com/2026/11/27/ai-agents-round/</link>
  <description><![CDATA[<p>Une startup lève <b>200M$</b> pour ses agents IA.</p>]]></description>
  <pubDate>Thu, 27 Nov 2026 10:00:00 +0000</pubDate>
</item>
<item>
  <title>Old news from last month</title>
  <link>https://techcrunch.com/2026/10/01/old/</link>
  <description>too old</description>
  <pubDate>Wed, 01 Oct 2026 10:00:00 +0000</pubDate>
</item>
</channel></rss>
"""

FAKE_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Cyber</title>
<entry>
  <title>Zero-day patched in widely used library</title>
  <link href="https://example-cyber.com/zero-day"/>
  <summary>Une zero-day corrigée.</summary>
  <published>2026-11-27T09:30:00Z</published>
</entry>
</feed>
"""

FAKE_DIGEST = {
    "subject": "Veille tech du jour",
    "email_body": (
        "Intro.\n\n### New AI chip announced by a major vendor\nRésumé.\n\n"
        "### Startup raises big round for AI agents\nRésumé.\n\n"
        "### Zero-day patched in widely used library\nRésumé."
    ),
    "podcast_script": (
        "Voici les principales actualités tech du jour, entre Hacker News et la presse spécialisée. "
        "Un fondeur annonce une nouvelle puce dédiée à l'IA. "
        "Une startup lève deux cents millions pour ses agents IA. "
        "Une zero-day est corrigée dans une bibliothèque très répandue. Bonne écoute."
    ),
    "stories": [
        {
            "title": "New AI chip announced by a major vendor",
            "url": "https://example.com/chip",
            "summary": "Résumé.",
            "source": "Hacker News",
            "category": "ia",
            "points": 500,
            "num_comments": 200,
        },
        {
            "title": "Startup raises big round for AI agents",
            "url": "https://techcrunch.com/2026/11/27/ai-agents-round/",
            "summary": "Résumé.",
            "source": "TechCrunch",
            "category": "ia",
            "points": 0,
            "num_comments": 0,
        },
    ],
}


async def _fake_http_get_json(url, params):
    assert params["tags"] == "front_page"
    return FAKE_HITS


async def _fake_http_get_text(url):
    if "techcrunch.com" in url:
        return FAKE_RSS
    if "thehackernews.com" in url:
        return FAKE_ATOM
    raise RuntimeError(f"feed en panne: {url}")


async def _fake_chat_complete_json(system, user):
    assert "secteur technologique" in system
    assert "top_n = 10" in user
    assert '"Hacker News"' in user
    assert "TechCrunch" in user
    return FAKE_DIGEST


async def _fake_tts_mp3(script, voice_id):
    assert voice_id == mod.DEFAULT_VOICE_ID
    assert "actualités tech" in script
    return "QUJD" * 64


def test_parse_json_object_plain() -> None:
    assert mod._parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_wrapped() -> None:
    assert mod._parse_json_object('Voici le JSON :\n{"a": 1}\nfin') == {"a": 1}


def test_extract_message_id_dict() -> None:
    assert mod._extract_message_id({"id": "msg-1", "threadId": "t"}) == "msg-1"


def test_extract_message_id_content_blocks() -> None:
    result = {"content": [{"type": "text", "text": '{"id": "msg-2"}'}]}
    assert mod._extract_message_id(result) == "msg-2"
    assert mod._extract_message_id({"content": []}) is None


def test_extract_own_email_threads_dict() -> None:
    result = {
        "threads": [
            {
                "id": "t1",
                "messages": [
                    {
                        "id": "m1",
                        "sender": "Denis <dyonysosd@gmail.com>",
                        "subject": "hi",
                    }
                ],
            }
        ]
    }
    assert mod._extract_own_email(result) == "dyonysosd@gmail.com"


def test_extract_own_email_content_blocks() -> None:
    inner = {"threads": [{"messages": [{"sender": "me@example.com"}]}]}
    result = {"content": [{"type": "text", "text": json.dumps(inner)}]}
    assert mod._extract_own_email(result) == "me@example.com"
    assert mod._extract_own_email({"threads": []}) is None
    assert mod._extract_own_email({"content": []}) is None


def test_parse_feed_date_rfc822_and_iso() -> None:
    parsed = mod._parse_feed_date("Thu, 27 Nov 2026 10:00:00 +0000")
    assert parsed is not None and parsed.year == 2026 and parsed.tzinfo is not None
    assert mod._parse_feed_date("2026-11-27T09:30:00Z") is not None
    assert mod._parse_feed_date("not a date") is None
    assert mod._parse_feed_date("") is None


def test_parse_feed_rss_filters_old_and_strips_html() -> None:
    items = mod._parse_feed("TechCrunch", FAKE_RSS, NOW)
    assert len(items) == 1
    assert items[0].title == "Startup raises big round for AI agents"
    assert items[0].source == "TechCrunch"
    assert items[0].url == "https://techcrunch.com/2026/11/27/ai-agents-round/"
    assert "200M$" in items[0].summary and "<b>" not in items[0].summary
    assert items[0].published.startswith("2026-11-27T10:00:00")


def test_parse_feed_atom_entry() -> None:
    items = mod._parse_feed("The Hacker News", FAKE_ATOM, NOW)
    assert len(items) == 1
    assert items[0].title == "Zero-day patched in widely used library"
    assert items[0].url == "https://example-cyber.com/zero-day"
    assert items[0].published.startswith("2026-11-27T09:30:00")


def test_parse_feed_caps_items_per_feed() -> None:
    entry = (
        "<item><title>t{i}</title><link>https://x/{i}</link>"
        "<pubDate>Thu, 27 Nov 2026 10:00:00 +0000</pubDate></item>"
    )
    xml = (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>f</title>'
        + "".join(entry.format(i=i) for i in range(20))
        + "</channel></rss>"
    )
    items = mod._parse_feed("f", xml, NOW)
    assert len(items) == mod.RSS_MAX_PER_FEED


async def test_fetch_one_feed_tolerates_broken_feed(monkeypatch) -> None:
    async def broken(url):
        raise RuntimeError("feed en panne")

    monkeypatch.setattr(mod, "_http_get_text", broken)
    items, ok = await mod._fetch_one_feed("TechCrunch", "https://x/feed", NOW)
    assert items == [] and ok is False


def test_aggregate_feeds_dedupes_and_counts() -> None:
    rss_items = mod._parse_feed("TechCrunch", FAKE_RSS, NOW)
    atom_items = mod._parse_feed("The Hacker News", FAKE_ATOM, NOW)
    dupe = [
        mod.NewsItem(source="Wired", title=rss_items[0].title, url=rss_items[0].url)
    ]
    result = mod._aggregate_feeds(
        [(rss_items, True), (atom_items, True), (dupe, True), ([], False)]
    )
    assert result.feeds_ok == 3
    assert len(result.items) == 2
    urls = {item.url for item in result.items}
    assert "https://techcrunch.com/2026/11/27/ai-agents-round/" in urls
    assert "https://example-cyber.com/zero-day" in urls


async def test_hn_tech_digest_end_to_end(monkeypatch, temporal_env) -> None:
    sent: list[dict] = []

    async def fake_gmail_send(client, *, to, subject, body, audio_b64, filename):
        sent.append(
            {
                "to": to,
                "subject": subject,
                "body": body,
                "audio_b64": audio_b64,
                "filename": filename,
            }
        )
        return {"content": [{"type": "text", "text": '{"id": "msg-123"}'}]}

    async def fake_gmail_search(client, query, page_size):
        assert query.startswith("from:me")
        return {"threads": [{"messages": [{"sender": "dyonysosd@gmail.com"}]}]}

    monkeypatch.setattr(mod, "_http_get_json", _fake_http_get_json)
    monkeypatch.setattr(mod, "_http_get_text", _fake_http_get_text)
    monkeypatch.setattr(mod, "_chat_complete_json", _fake_chat_complete_json)
    monkeypatch.setattr(mod, "_tts_mp3", _fake_tts_mp3)
    monkeypatch.setattr(mod, "_gmail_send_email", fake_gmail_send)
    monkeypatch.setattr(mod, "_gmail_search_threads", fake_gmail_search)

    async with create_test_worker(
        temporal_env,
        workflows=[mod.HnTechDigestWorkflow],
        activities=[
            mod.resolve_own_email,
            mod.fetch_front_page,
            mod.fetch_news_feeds,
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

        handle2 = await temporal_env.client.start_workflow(
            "hn-tech-digest",
            {"to": "custom@example.com"},
            id="test-hn-tech-digest-override",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result2 = await asyncio.wait_for(handle2.result(), timeout=15)

    assert result["message_id"] == "msg-123"
    assert result["subject"] == "Veille tech du jour"
    assert result["stories_count"] == 4
    assert result["articles_count"] == 2
    assert result["audio_b64_chars"] == len("QUJD" * 64)
    assert sent[0]["to"] == "dyonysosd@gmail.com"
    assert sent[1]["to"] == "custom@example.com"
    assert result2["message_id"] == "msg-123"
    for entry in sent:
        assert entry["filename"].startswith("hn-tech-digest-")
        assert entry["filename"].endswith(".mp3")
        assert entry["audio_b64"] == "QUJD" * 64
        assert "New AI chip announced by a major vendor" in entry["body"]
