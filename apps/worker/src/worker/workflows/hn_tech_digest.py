"""One-off tech digest: fetch the Hacker News front page plus tech-news RSS feeds
(cybersecurity, AI, general tech), keep only tech-sector news, then email a French
summary with a short podcast-style MP3 to the triggering user's own Gmail inbox.

On-behalf-of workflow: manual runs only, never scheduled (the platform rejects
schedules on on_behalf_of workflows). The gmail connector and the Mistral API
(chat + Voxtral TTS) run with the triggering user's credentials, so no secrets
are stored in the repo or bound to the deployment.
"""

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta

import pydantic
from mistralai import workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai import get_mistral_client
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail = connector("gmail")

HN_FRONT_PAGE_URL = "https://hn.algolia.com/api/v1/search"
HN_HITS_PER_PAGE = 30
NEWS_FEEDS = (
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Wired", "https://www.wired.com/feed/rss"),
    ("VentureBeat", "https://venturebeat.com/feed/"),
    ("ZDNet", "https://www.zdnet.com/news/rss.xml"),
    ("BleepingComputer", "https://www.bleepingcomputer.com/feed/"),
    ("Krebs on Security", "https://krebsonsecurity.com/feed/"),
    ("The Hacker News", "https://thehackernews.com/feeds/posts/default"),
    ("Numerama", "https://www.numerama.com/feed/"),
)
RSS_MAX_PER_FEED = 5
RSS_MAX_AGE_HOURS = 72
CHAT_MODEL = "mistral-medium-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_RECIPIENT = "dsanchez@campus-eni.fr"
DEFAULT_VOICE_ID = "voice_592"
DEFAULT_TOP_N = 10
MAX_ATTACHMENT_B64_CHARS = 4_000_000

SYSTEM_PROMPT = """Tu es un journaliste spécialisé dans l'actualité technologique.
On te fournit les stories de la page d'accueil de Hacker News et les derniers articles de flux RSS d'actualités tech (cybersécurité, IA, produits tech). Chaque entrée JSON a un champ "source" : "Hacker News" pour les stories HN (avec "points" et "num_comments"), ou le nom du média pour les articles RSS (avec "published"). On te donne aussi une limite top_n.

Ta mission :
1. Sélectionne UNIQUEMENT les actualités du secteur technologique : logiciel, matériel, intelligence artificielle, cybersécurité, startups tech, produits et services tech, open source majeur, standards web et navigateurs, cloud et infrastructures, mobile, semi-conducteurs, régulation du secteur tech.
2. EXCLUS strictement : blogs personnels, essais et opinions personnelles, "Ask HN", sondages, actualités non tech (politique, faits divers, sciences non technologiques, sport, culture), et doublons (le même sujet traité par plusieurs sources ne compte qu'une fois).
3. Classe les actualités retenues par importance (pertinence tech, points, commentaires, récence) et garde au plus top_n actualités.
4. Pour chaque actualité retenue, choisis une "category" parmi "ia", "cybersécurité", "tech".
5. Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte autour, avec exactement ces champs :
   - "subject" : objet de l'email en français, court.
   - "email_body" : corps de l'email en Markdown, en français : une courte intro, puis pour chaque actualité retenue un bloc "### <titre>" suivi d'un résumé en 1-2 phrases en français, puis une ligne "Source : <source> — <url>".
   - "podcast_script" : script audio façon podcast en français, entre 150 et 220 mots (environ 1 à 2 minutes à l'oral) : commence par une intro du type "Voici les principales actualités tech du jour, entre Hacker News et la presse spécialisée", présente chaque titre retenu en une ou deux phrases, termine par une courte conclusion. Style oral et dynamique, sans markdown.
   - "stories" : liste des actualités retenues, chacune avec "title", "url", "summary" (1-2 phrases en français), "source", "category", "points", "num_comments" (0 quand absent).
Si aucune actualité n'est tech, "stories" est vide, "email_body" l'indique en une phrase et "podcast_script" est une annonce de moins de 60 mots."""


class HnStory(pydantic.BaseModel):
    object_id: str
    title: str
    url: str | None = None
    author: str = ""
    points: int = 0
    num_comments: int = 0


class NewsItem(pydantic.BaseModel):
    source: str
    title: str
    url: str | None = None
    summary: str = ""
    published: str = ""


class FeedsResult(pydantic.BaseModel):
    items: list[NewsItem] = pydantic.Field(default_factory=list)
    feeds_ok: int = 0


class KeptStory(pydantic.BaseModel):
    title: str
    url: str | None = None
    summary: str = ""
    source: str = ""
    category: str = ""
    points: int = 0
    num_comments: int = 0


class DigestContent(pydantic.BaseModel):
    subject: str
    email_body: str
    podcast_script: str
    stories: list[KeptStory] = pydantic.Field(default_factory=list)


class DigestInput(pydantic.BaseModel):
    to: str = ""  # empty = the triggering user's own Gmail address, resolved at runtime
    top_n: int = DEFAULT_TOP_N
    voice_id: str = DEFAULT_VOICE_ID


class DigestResult(pydantic.BaseModel):
    message_id: str | None
    subject: str
    stories_count: int
    articles_count: int
    audio_b64_chars: int


async def _http_get_json(url: str, params: dict) -> dict:
    import httpx

    async with httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": "hn-tech-digest-workflow/1.0"}
    ) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def _http_get_text(url: str) -> str:
    import httpx

    async with httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": "hn-tech-digest-workflow/1.0"}
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def _parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _parse_feed_date(value: str) -> datetime | None:
    from email.utils import parsedate_to_datetime

    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return (child.text or "").strip()
    return ""


def _parse_feed(source: str, xml_text: str, now: datetime) -> list[NewsItem]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    entries = [
        child for child in root.iter() if _local_name(child.tag) in ("item", "entry")
    ]
    max_age = timedelta(hours=RSS_MAX_AGE_HOURS)
    items: list[NewsItem] = []
    for entry in entries[: RSS_MAX_PER_FEED * 3]:
        title = _child_text(entry, "title")
        if not title:
            continue
        url = _child_text(entry, "link")
        if not url:
            for child in entry:
                if _local_name(child.tag) == "link" and child.get("href"):
                    url = child.get("href", "").strip()
                    break
        summary = _strip_html(
            _child_text(entry, "description")
            or _child_text(entry, "summary")
            or _child_text(entry, "content")
        )
        raw_date = (
            _child_text(entry, "pubdate")
            or _child_text(entry, "published")
            or _child_text(entry, "updated")
            or _child_text(entry, "date")
        )
        published = _parse_feed_date(raw_date)
        if published is not None and now - published > max_age:
            continue
        items.append(
            NewsItem(
                source=source,
                title=title,
                url=url or None,
                summary=summary[:500],
                published=published.isoformat() if published else raw_date,
            )
        )
        if len(items) >= RSS_MAX_PER_FEED:
            break
    return items


async def _fetch_one_feed(
    source: str, url: str, now: datetime
) -> tuple[list[NewsItem], bool]:
    try:
        xml_text = await _http_get_text(url)
        return _parse_feed(source, xml_text, now), True
    except Exception:  # noqa: BLE001  one broken feed must not fail the whole fetch
        return [], False


def _aggregate_feeds(results: list[tuple[list[NewsItem], bool]]) -> FeedsResult:
    seen: set[str] = set()
    items: list[NewsItem] = []
    feeds_ok = 0
    for feed_items, ok in results:
        feeds_ok += int(ok)
        for item in feed_items:
            key = (item.url or item.title).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return FeedsResult(items=items, feeds_ok=feeds_ok)


async def _chat_complete_json(system: str, user: str) -> dict:
    client = get_mistral_client()
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return _parse_json_object(content)


async def _tts_mp3(script: str, voice_id: str) -> str:
    client = get_mistral_client()
    response = await client.audio.speech.complete_async(
        model=TTS_MODEL,
        input=script,
        voice_id=voice_id,
        response_format="mp3",
    )
    return response.audio_data


async def _gmail_send_email(
    client: ToolCallClient,
    *,
    to: str,
    subject: str,
    body: str,
    audio_b64: str,
    filename: str,
) -> object:
    return await client.call_tool(
        tool_name="send_email",
        arguments={
            "to": [to],
            "subject": subject,
            "body": body,
            "attachments": [
                {
                    "filename": filename,
                    "mimeType": "audio/mpeg",
                    "content": audio_b64,
                    "contentEncoding": "base64",
                }
            ],
        },
    )


def _extract_message_id(result: object) -> str | None:
    if isinstance(result, dict):
        candidate = result.get("id") or result.get("messageId")
        if candidate:
            return str(candidate)
        result = result.get("content")
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    parsed = json.loads(block.get("text") or "")
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    candidate = parsed.get("id") or parsed.get("messageId")
                    if candidate:
                        return str(candidate)
    return None


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _extract_own_email(result: object) -> str | None:
    if isinstance(result, dict):
        threads = result.get("threads")
        if isinstance(threads, list):
            for thread in threads:
                if not isinstance(thread, dict):
                    continue
                messages = thread.get("messages")
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    if isinstance(message, dict) and message.get("sender"):
                        match = _EMAIL_RE.search(str(message["sender"]))
                        if match:
                            return match.group(0)
        result = result.get("content")
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    parsed = json.loads(block.get("text") or "")
                except json.JSONDecodeError:
                    continue
                found = _extract_own_email(parsed)
                if found:
                    return found
    return None


async def _gmail_search_threads(
    client: ToolCallClient, query: str, page_size: int
) -> object:
    return await client.call_tool(
        tool_name="search_threads",
        arguments={"query": query, "pageSize": page_size},
    )


@workflows.activity(
    name="hn-fetch-front-page",
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=3,
)
async def fetch_front_page() -> list[HnStory]:
    payload = await _http_get_json(
        HN_FRONT_PAGE_URL, {"tags": "front_page", "hitsPerPage": HN_HITS_PER_PAGE}
    )
    stories: list[HnStory] = []
    for hit in payload.get("hits") or []:
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        stories.append(
            HnStory(
                object_id=str(hit.get("objectID") or ""),
                title=title,
                url=hit.get("url"),
                author=hit.get("author") or "",
                points=int(hit.get("points") or 0),
                num_comments=int(hit.get("num_comments") or 0),
            )
        )
    return stories


@workflows.activity(
    name="hn-fetch-news-feeds",
    start_to_close_timeout=timedelta(seconds=120),
    retry_policy_max_attempts=2,
)
async def fetch_news_feeds() -> FeedsResult:
    now = datetime.now(UTC)
    results = await asyncio.gather(
        *(_fetch_one_feed(source, url, now) for source, url in NEWS_FEEDS)
    )
    return _aggregate_feeds(results)


def _digest_user_payload(
    stories: list[HnStory], articles: list[NewsItem], top_n: int
) -> str:
    lines = [
        json.dumps(
            {
                "source": "Hacker News",
                "object_id": s.object_id,
                "title": s.title,
                "url": s.url,
                "author": s.author,
                "points": s.points,
                "num_comments": s.num_comments,
            },
            ensure_ascii=False,
        )
        for s in stories
    ]
    lines += [
        json.dumps(
            {
                "source": a.source,
                "title": a.title,
                "url": a.url,
                "summary": a.summary,
                "published": a.published,
            },
            ensure_ascii=False,
        )
        for a in articles
    ]
    return f"top_n = {top_n}\n\nentrées (JSON, une par ligne) :\n" + "\n".join(lines)


@workflows.activity(
    name="hn-build-digest",
    start_to_close_timeout=timedelta(seconds=180),
    retry_policy_max_attempts=2,
)
async def build_digest(
    stories: list[HnStory], articles: list[NewsItem], top_n: int
) -> DigestContent:
    payload = await _chat_complete_json(
        SYSTEM_PROMPT, _digest_user_payload(stories, articles, top_n)
    )
    return DigestContent.model_validate(payload)


@workflows.activity(
    name="hn-synthesize-podcast",
    start_to_close_timeout=timedelta(seconds=240),
    retry_policy_max_attempts=2,
)
async def synthesize_podcast(script: str, voice_id: str) -> str:
    audio_b64 = await _tts_mp3(script, voice_id)
    if len(audio_b64) > MAX_ATTACHMENT_B64_CHARS:
        raise ValueError(
            f"audio trop volumineux pour une pièce jointe Gmail: {len(audio_b64)} caractères base64"
        )
    return audio_b64


@workflows.activity(
    name="hn-resolve-own-email",
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=2,
)
async def resolve_own_email(
    gmail_client: ToolCallClient = Depends(gmail),  # noqa: B008  documented Depends() DI pattern
) -> str:
    result = await _gmail_search_threads(gmail_client, "from:me newer_than:365d", 1)
    return _extract_own_email(result) or DEFAULT_RECIPIENT


@workflows.activity(
    name="hn-send-digest-email",
    start_to_close_timeout=timedelta(seconds=120),
    retry_policy_max_attempts=1,
)
async def send_digest_email(
    to: str,
    subject: str,
    body: str,
    audio_b64: str,
    filename: str,
    gmail_client: ToolCallClient = Depends(gmail),  # noqa: B008  documented Depends() DI pattern
) -> str | None:
    result = await _gmail_send_email(
        gmail_client,
        to=to,
        subject=subject,
        body=body,
        audio_b64=audio_b64,
        filename=filename,
    )
    return _extract_message_id(result)


@workflows.workflow.define(
    name="hn-tech-digest",
    on_behalf_of=True,
    workflow_description=(
        "Veille tech à la demande (Hacker News + flux RSS tech/cyber/IA) : résumé en français "
        "envoyé dans Gmail avec un podcast audio MP3 de 1-2 min. Déclenchement manuel "
        "uniquement, jamais planifié."
    ),
)
@uses_connectors(gmail)
class HnTechDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, prompt: DigestInput) -> DigestResult:
        to = prompt.to.strip() or await resolve_own_email()
        stories, feeds = await asyncio.gather(fetch_front_page(), fetch_news_feeds())
        digest = await build_digest(stories, feeds.items, prompt.top_n)
        audio_b64 = await synthesize_podcast(digest.podcast_script, prompt.voice_id)
        filename = f"hn-tech-digest-{workflows.workflow.now().strftime('%Y-%m-%d')}.mp3"
        message_id = await send_digest_email(
            to, digest.subject, digest.email_body, audio_b64, filename
        )
        return DigestResult(
            message_id=message_id,
            subject=digest.subject,
            stories_count=len(stories),
            articles_count=len(feeds.items),
            audio_b64_chars=len(audio_b64),
        )
