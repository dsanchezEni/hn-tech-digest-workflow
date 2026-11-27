"""GitHub repo catch-up: given a repository (owner/repo) and a date, list the pull
requests merged since that date and the ones still open awaiting review, summarize
each in one line (in French) with its link, and post the recap to the triggering
user's own Microsoft Teams chat (the one-on-one chat with themselves).

On-behalf-of workflow: manual runs only, never scheduled (the platform rejects
schedules on on_behalf_of workflows). The github_app and teams connectors plus the
Mistral API calls run with the triggering user's credentials, so no secrets are
stored in the repo or bound to the deployment. Pass dry_run=true to build and return
the recap without posting it to Teams.
"""

import asyncio
import json
import re
from datetime import timedelta

import pydantic
from mistralai import workflows
from mistralai.client.models import TextChunk
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai import Agent, Runner, get_mistral_client
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

github = connector("github_app")
teams = connector("teams")

CHAT_MODEL = "mistral-medium-latest"
PR_FIELDS = ["number", "title", "body", "html_url", "user", "closed_at", "labels"]
PR_SEARCH_PAGE_SIZE = 100
MAX_PR_BODY_CHARS = 1500
MAX_PRS_PER_SECTION = 50
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SUMMARIZE_SYSTEM_PROMPT = """Tu es un assistant qui rédige des résumés de pull requests GitHub en français.
On te donne deux listes de PR (fusionnées depuis une date, et ouvertes en attente de revue), au format JSON une par ligne, avec number, title, author, labels, body (tronqué) et url.

Ta mission : pour CHAQUE PR des deux listes, rédiger un "one_line" : un résumé en UNE SEULE ligne, en français, factuel (quoi, et pourquoi/impact si visible dans le titre ou le body), sans répéter le titre en entier, sans markdown, sans URL.

Réponds UNIQUEMENT avec un objet JSON valide, sans aucun texte autour :
{"merged": {"<number>": "<one_line>", ...}, "pending": {"<number>": "<one_line>", ...}}
Les clés sont les numéros de PR (chaînes de caractères). Si une liste est vide, son objet est vide. N'invente aucune PR."""

TEAMS_AGENT_INSTRUCTIONS = """Tu as accès aux outils du connecteur Microsoft Teams de l'utilisateur.
Ta mission : publier le message fourni ci-dessous dans la conversation Teams de l'utilisateur AVEC LUI-MÊME (le chat oneOnOne dont il est le seul membre, c'est-à-dire le chat avec soi-même).
1. Liste les chats de l'utilisateur et identifie ce chat avec lui-même.
2. Envoie le message tel quel dans ce chat, sans le modifier.
3. Réponds uniquement par une confirmation courte : l'ID du chat et l'ID du message envoyé.
Ne poste dans aucun autre chat ou canal, ne modifie ni ne supprime aucun message."""


class CatchupInput(pydantic.BaseModel):
    repo: str
    since: str
    dry_run: bool = False


class PrSummary(pydantic.BaseModel):
    number: int
    title: str
    url: str
    one_line: str


class CatchupReport(pydantic.BaseModel):
    merged: list[PrSummary] = pydantic.Field(default_factory=list)
    pending: list[PrSummary] = pydantic.Field(default_factory=list)


class CatchupResult(pydantic.BaseModel):
    repo: str
    since: str
    merged_count: int
    pending_count: int
    message: str
    teams_posted: bool
    teams_confirmation: str


def _parse_repo(value: str) -> tuple[str, str]:
    parts = [part for part in value.strip().split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"repo attendue au format 'owner/repo', reçu : {value!r}")
    return parts[0], parts[1]


def _validate_since(value: str) -> str:
    value = value.strip()
    if not _DATE_RE.match(value):
        raise ValueError(f"since attendue au format YYYY-MM-DD, reçu : {value!r}")
    return value


def _tool_payload(result: object) -> object:
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        result = result["content"]
    if isinstance(result, list):
        texts = [
            block.get("text", "")
            for block in result
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        raw = "\n".join(texts)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return result


def _pr_items(result: object) -> list[dict]:
    payload = _tool_payload(result)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return [item for item in payload["items"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _normalize_pr(item: dict) -> dict:
    user = item.get("user") if isinstance(item.get("user"), dict) else {}
    labels = [
        label.get("name", "")
        for label in (item.get("labels") or [])
        if isinstance(label, dict)
    ]
    body = (item.get("body") or "").strip()
    return {
        "number": int(item.get("number") or 0),
        "title": (item.get("title") or "").strip(),
        "url": item.get("html_url") or "",
        "author": user.get("login", ""),
        "closed_at": item.get("closed_at") or "",
        "labels": [label for label in labels if label],
        "body": body[:MAX_PR_BODY_CHARS],
    }


async def _github_search_prs(
    client: ToolCallClient, owner: str, repo: str, query: str
) -> list[dict]:
    result = await client.call_tool(
        tool_name="search_pull_requests",
        arguments={
            "query": query,
            "owner": owner,
            "repo": repo,
            "fields": PR_FIELDS,
            "page": 1,
            "perPage": PR_SEARCH_PAGE_SIZE,
            "sort": "updated",
            "order": "desc",
        },
    )
    return [_normalize_pr(item) for item in _pr_items(result)]


def _summarize_user_payload(repo: str, since: str, merged: list[dict], pending: list[dict]) -> str:
    def section(name: str, prs: list[dict]) -> str:
        lines = [
            json.dumps(
                {
                    "number": pr["number"],
                    "title": pr["title"],
                    "author": pr["author"],
                    "labels": pr["labels"],
                    "closed_at": pr["closed_at"],
                    "body": pr["body"],
                    "url": pr["url"],
                },
                ensure_ascii=False,
            )
            for pr in prs[:MAX_PRS_PER_SECTION]
        ]
        return f"## {name} ({len(prs)})\n" + "\n".join(lines)

    return (
        f"repo = {repo}\nsince = {since}\n\n"
        + section("merged", merged)
        + "\n\n"
        + section("pending", pending)
    )


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
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _build_summaries(prs: list[dict], raw: object) -> list[PrSummary]:
    summaries = raw if isinstance(raw, dict) else {}
    out: list[PrSummary] = []
    for pr in prs[:MAX_PRS_PER_SECTION]:
        one_line = str(summaries.get(str(pr["number"]), "")).strip() or pr["title"]
        out.append(
            PrSummary(
                number=pr["number"],
                title=pr["title"],
                url=pr["url"],
                one_line=one_line,
            )
        )
    return out


def _render_message(report: CatchupReport, repo: str, since: str, generated_at: str) -> str:
    lines = [
        f"**Rattrapage GitHub — `{repo}`**",
        f"Depuis le **{since}** — généré le {generated_at}",
        "",
        f"## ✅ Fusionnés depuis le {since} ({len(report.merged)})",
    ]
    if report.merged:
        for pr in report.merged:
            lines.append(f"- **#{pr.number} {pr.title}** — {pr.one_line} — {pr.url}")
    else:
        lines.append("_Aucun pull request fusionné depuis cette date._")
    lines += ["", f"## ⏳ En attente de revue ({len(report.pending)})"]
    if report.pending:
        for pr in report.pending:
            lines.append(f"- **#{pr.number} {pr.title}** — {pr.one_line} — {pr.url}")
    else:
        lines.append("_Aucun pull request ouvert en attente de revue._")
    return "\n".join(lines)


async def _run_teams_agent(message: str) -> str:
    agent = Agent(
        name="teams-self-chat-poster",
        model=CHAT_MODEL,
        instructions=TEAMS_AGENT_INSTRUCTIONS,
        connectors=[teams],
    )
    outputs = await Runner.run(agent=agent, inputs=message, max_turns=10)
    return "\n".join(output.text for output in outputs if isinstance(output, TextChunk))


@workflows.activity(
    name="gh-fetch-merged-prs",
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=3,
)
async def fetch_merged_prs(
    owner: str,
    repo: str,
    since: str,
    github_client: ToolCallClient = Depends(github),  # noqa: B008  documented Depends() DI pattern
) -> list[dict]:
    return await _github_search_prs(github_client, owner, repo, f"is:merged merged:>={since}")


@workflows.activity(
    name="gh-fetch-open-prs",
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy_max_attempts=3,
)
async def fetch_open_prs(
    owner: str,
    repo: str,
    github_client: ToolCallClient = Depends(github),  # noqa: B008  documented Depends() DI pattern
) -> list[dict]:
    return await _github_search_prs(github_client, owner, repo, "is:open -is:draft")


@workflows.activity(
    name="gh-summarize-prs",
    start_to_close_timeout=timedelta(seconds=180),
    retry_policy_max_attempts=2,
)
async def summarize_prs(
    repo: str, since: str, merged: list[dict], pending: list[dict]
) -> CatchupReport:
    payload = await _chat_complete_json(
        SUMMARIZE_SYSTEM_PROMPT, _summarize_user_payload(repo, since, merged, pending)
    )
    return CatchupReport(
        merged=_build_summaries(merged, payload.get("merged")),
        pending=_build_summaries(pending, payload.get("pending")),
    )


@workflows.activity(
    name="teams-post-catchup",
    start_to_close_timeout=timedelta(minutes=10),
    retry_policy_max_attempts=1,
)
async def post_to_teams(message: str) -> str:
    return await _run_teams_agent(message)


@workflows.workflow.define(
    name="github-repo-catchup",
    on_behalf_of=True,
    workflow_description=(
        "Rattrapage d'un dépôt GitHub depuis une date : PR fusionnées et PR en attente de "
        "revue, chacune avec un résumé en une ligne et un lien, publié dans la conversation "
        "Teams de l'utilisateur avec lui-même. Déclenchement manuel uniquement, jamais planifié."
    ),
)
@uses_connectors(github, teams)
class GithubRepoCatchupWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, prompt: CatchupInput) -> CatchupResult:
        owner, repo = _parse_repo(prompt.repo)
        since = _validate_since(prompt.since)
        merged, pending = await asyncio.gather(
            fetch_merged_prs(owner, repo, since),
            fetch_open_prs(owner, repo),
        )
        report = await summarize_prs(prompt.repo, since, merged, pending)
        generated_at = workflows.workflow.now().strftime("%Y-%m-%d %H:%M UTC")
        message = _render_message(report, prompt.repo, since, generated_at)
        if prompt.dry_run:
            return CatchupResult(
                repo=prompt.repo,
                since=since,
                merged_count=len(report.merged),
                pending_count=len(report.pending),
                message=message,
                teams_posted=False,
                teams_confirmation="dry_run=true : récapitulatif généré mais non posté sur Teams",
            )
        confirmation = await post_to_teams(message)
        return CatchupResult(
            repo=prompt.repo,
            since=since,
            merged_count=len(report.merged),
            pending_count=len(report.pending),
            message=message,
            teams_posted=True,
            teams_confirmation=confirmation,
        )
