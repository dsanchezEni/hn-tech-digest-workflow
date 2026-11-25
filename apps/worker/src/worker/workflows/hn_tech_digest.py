"""HackerNews tech digest, run on a daily schedule (and manually on demand).

Fetches the Hacker News front page, keeps only tech-sector news, then emails a
French summary with a short podcast-style MP3 to the configured Gmail inbox.

The workflow runs as the deployment's service identity (not on-behalf-of), so it
can be scheduled: Mistral chat/TTS calls use the worker's credentials, and the
email is sent through the Gmail REST API using a refresh token bound from
workspace secrets (GMAIL_CLIENT_ID / GMAIL_CLIENT_SECRET / GMAIL_REFRESH_TOKEN).
"""

import base64
import json
import os
import re
from datetime import timedelta

import pydantic
from mistralai import workflows
from mistralai.workflows.plugins.mistralai import get_mistral_client

HN_FRONT_PAGE_URL = "https://hn.algolia.com/api/v1/search"
HN_HITS_PER_PAGE = 30
CHAT_MODEL = "mistral-medium-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
DEFAULT_RECIPIENT = "dsanchez@campus-eni.fr"
DEFAULT_VOICE_ID = "voice_592"
DEFAULT_TOP_N = 10
MAX_ATTACHMENT_B64_CHARS = 4_000_000
GMAIL_ENV_VARS = ("GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET", "GMAIL_REFRESH_TOKEN")

SYSTEM_PROMPT = """Tu es un journaliste spécialisé dans l'actualité technologique.
On te fournit les stories de la page d'accueil de Hacker News (object_id, titre, url, auteur, points, num_comments) et une limite top_n.

Ta mission :
1. Sélectionne UNIQUEMENT les actualités du secteur technologique : logiciel, matériel, intelligence artificielle, cybersécurité, startups tech, produits et services tech, open source majeur, standards web et navigateurs, cloud et infrastructures, mobile, semi-conducteurs, régulation du secteur tech.
2. EXCLUS strictement : blogs personnels, essais et opinions personnelles, "Ask HN", sondages, actualités non tech (politique, faits divers, sciences non technologiques, sport, culture), et doublons.
3. Classe les stories retenues par importance (pertinence tech, points, commentaires) et garde au plus top_n stories.
4. Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte autour, avec exactement ces champs :
   - "subject" : objet de l'email en français, court.
   - "email_body" : corps de l'email en Markdown, en français : une courte intro, puis pour chaque story retenue un bloc "### <titre>" suivi d'un résumé en 1-2 phrases en français, puis une ligne "Source : <url> — <points> points, <num_comments> commentaires".
   - "podcast_script" : script audio façon podcast en français, entre 150 et 220 mots (environ 1 à 2 minutes à l'oral) : commence par une intro du type "Voici les principales actualités tech du jour sur Hacker News", présente chaque titre retenu en une ou deux phrases, termine par une courte conclusion. Style oral et dynamique, sans markdown.
   - "stories" : liste des stories retenues, chacune avec "title", "url", "summary" (1-2 phrases en français), "points", "num_comments".
Si aucune story n'est une actualité tech, "stories" est vide, "email_body" l'indique en une phrase et "podcast_script" est une annonce de moins de 60 mots."""


class HnStory(pydantic.BaseModel):
    object_id: str
    title: str
    url: str | None = None
    author: str = ""
    points: int = 0
    num_comments: int = 0


class KeptStory(pydantic.BaseModel):
    title: str
    url: str | None = None
    summary: str = ""
    points: int = 0
    num_comments: int = 0


class DigestContent(pydantic.BaseModel):
    subject: str
    email_body: str
    podcast_script: str
    stories: list[KeptStory] = pydantic.Field(default_factory=list)


class DigestInput(pydantic.BaseModel):
    to: str = DEFAULT_RECIPIENT
    top_n: int = DEFAULT_TOP_N
    voice_id: str = DEFAULT_VOICE_ID


class DigestResult(pydantic.BaseModel):
    message_id: str | None
    subject: str
    stories_count: int
    audio_b64_chars: int


async def _http_get_json(url: str, params: dict) -> dict:
    import httpx

    async with httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": "hn-tech-digest-workflow/1.0"}
    ) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def _parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


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


def _build_raw_mime(
    to: str, subject: str, body: str, audio_b64: str, filename: str
) -> str:
    from email.message import EmailMessage

    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    message.add_attachment(
        base64.b64decode(audio_b64), maintype="audio", subtype="mpeg", filename=filename
    )
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


async def _gmail_access_token() -> str:
    import httpx

    missing = [name for name in GMAIL_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Gmail workspace secrets not bound to the worker: {', '.join(missing)}"
        )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": os.environ["GMAIL_CLIENT_ID"],
                "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
                "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]


async def _gmail_send_email(
    to: str, subject: str, body: str, audio_b64: str, filename: str
) -> str | None:
    import httpx

    token = await _gmail_access_token()
    raw = _build_raw_mime(to, subject, body, audio_b64, filename)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GMAIL_SEND_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"raw": raw},
        )
        response.raise_for_status()
        return response.json().get("id")


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


def _digest_user_payload(stories: list[HnStory], top_n: int) -> str:
    lines = [
        json.dumps(
            {
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
    return f"top_n = {top_n}\n\nstories (JSON, une par ligne) :\n" + "\n".join(lines)


@workflows.activity(
    name="hn-build-digest",
    start_to_close_timeout=timedelta(seconds=180),
    retry_policy_max_attempts=2,
)
async def build_digest(stories: list[HnStory], top_n: int) -> DigestContent:
    payload = await _chat_complete_json(
        SYSTEM_PROMPT, _digest_user_payload(stories, top_n)
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
    name="hn-send-digest-email",
    start_to_close_timeout=timedelta(seconds=120),
    retry_policy_max_attempts=1,
)
async def send_digest_email(
    to: str, subject: str, body: str, audio_b64: str, filename: str
) -> str | None:
    return await _gmail_send_email(to, subject, body, audio_b64, filename)


@workflows.workflow.define(
    name="hn-tech-digest",
    workflow_description=(
        "Veille tech Hacker News : résumé en français envoyé dans Gmail avec un podcast audio "
        "MP3 de 1-2 min. Planifié tous les jours à 08:00 Europe/Paris, aussi lançable manuellement."
    ),
)
class HnTechDigestWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, prompt: DigestInput) -> DigestResult:
        stories = await fetch_front_page()
        digest = await build_digest(stories, prompt.top_n)
        audio_b64 = await synthesize_podcast(digest.podcast_script, prompt.voice_id)
        filename = f"hn-tech-digest-{workflows.workflow.now().strftime('%Y-%m-%d')}.mp3"
        message_id = await send_digest_email(
            prompt.to, digest.subject, digest.email_body, audio_b64, filename
        )
        return DigestResult(
            message_id=message_id,
            subject=digest.subject,
            stories_count=len(digest.stories),
            audio_b64_chars=len(audio_b64),
        )
