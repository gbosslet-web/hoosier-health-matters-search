from __future__ import annotations

import json
import math
import os
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_PATH = DATA_DIR / "archive_index.json"
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_RSS_URL = os.getenv("HHM_RSS_URL", "https://feeds.buzzsprout.com/2446815.rss")
CACHE_VERSION = 2
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("OPENAI_RESPONSE_MODEL", "gpt-4.1-mini")
TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
OPENAI_CONFIGURED = bool(os.getenv("OPENAI_API_KEY"))
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


class PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self.parts)


def get_logo_path() -> Path:
    return ASSETS_DIR / "HHM logo.jpeg"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "episode"


def tokenize(value: str) -> list[str]:
    stop_words = {
        "a",
        "about",
        "an",
        "and",
        "did",
        "episode",
        "for",
        "in",
        "interview",
        "interviewed",
        "is",
        "it",
        "of",
        "on",
        "say",
        "said",
        "season",
        "the",
        "to",
        "was",
        "we",
        "what",
        "when",
        "who",
    }
    return [token for token in re.findall(r"[a-z0-9']+", value.lower()) if token not in stop_words]


def expand_keywords(query: str) -> list[str]:
    keywords = tokenize(query)
    lowered = query.lower()
    expansions: list[str] = []
    phrase_map = {
        "supreme court": ["scotus", "ruling", "rulings"],
        "medicaid reimbursement": ["medicaid", "reimbursement", "rates", "waiver"],
        "medicaid waivers": ["medicaid", "waiver", "rates"],
        "opioid deaths": ["opioid", "overdose"],
    }
    for phrase, related_terms in phrase_map.items():
        if phrase in lowered:
            expansions.extend(related_terms)
    return list(dict.fromkeys(keywords + expansions))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    denom_a = math.sqrt(sum(x * x for x in a))
    denom_b = math.sqrt(sum(y * y for y in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)


def mean_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    size = len(vectors[0])
    sums = [0.0] * size
    for vector in vectors:
        for index, value in enumerate(vector):
            sums[index] += value
    return [value / len(vectors) for value in sums]


def strip_html(value: str) -> str:
    parser = PlainTextHTMLParser()
    parser.feed(value or "")
    return normalize_space(unescape(parser.text()))


def parse_pub_date(value: str) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.date().isoformat(), parsed.isoformat()
    except (TypeError, ValueError):
        return None, None


def extract_episode_numbers(title: str, subtitle: str, summary: str) -> tuple[int | None, int | None]:
    haystack = " ".join(part for part in [title, subtitle, summary] if part)
    season_match = re.search(r"season\s+(\d+)", haystack, re.IGNORECASE)
    episode_match = re.search(r"episode\s+(\d+)", haystack, re.IGNORECASE)
    return (
        int(season_match.group(1)) if season_match else None,
        int(episode_match.group(1)) if episode_match else None,
    )


def looks_like_transcript(text: str) -> bool:
    if not text or len(text) < 500:
        return False
    lower = text.lower()
    transcript_markers = ["intro", "thanks for listening", "minute", "medicaid", "episode"]
    return sum(marker in lower for marker in transcript_markers) >= 2


def chunk_transcript(text: str, target_words: int = 180, overlap_words: int = 45) -> list[str]:
    words = text.split()
    if len(words) <= target_words:
        return [normalize_space(text)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk = normalize_space(" ".join(words[start:end]))
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def format_episode_label(episode: dict[str, Any]) -> str:
    title = episode.get("title") or "Untitled episode"
    season = episode.get("season")
    episode_number = episode.get("episode_number")
    if season and episode_number:
        return f"Season {season}, Episode {episode_number}: {title}"
    if episode_number:
        return f"Episode {episode_number}: {title}"
    return title


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_show_id(channel_link: str = "", rss_url: str = DEFAULT_RSS_URL) -> str:
    for source in [channel_link, rss_url]:
        match = re.search(r"(?:buzzsprout\.com/|feeds\.buzzsprout\.com/)(\d+)", source or "")
        if match:
            return match.group(1)
    return ""


def extract_episode_id(*values: str) -> str:
    for value in values:
        match = re.search(r"(\d{6,})", value or "")
        if match:
            return match.group(1)
    return ""


@lru_cache(maxsize=8)
def fetch_buzzsprout_episode_catalog(show_id: str) -> dict[str, str]:
    if not show_id:
        return {}
    catalog_url = f"https://hoosierhealthmatters.buzzsprout.com/{show_id}/episodes"
    try:
        html = fetch_bytes(catalog_url, timeout=30).decode("utf-8", errors="ignore")
    except Exception:
        return {}

    matches = re.findall(
        rf"https://hoosierhealthmatters\.buzzsprout\.com/{show_id}/episodes/\d+-[a-z0-9\-]+",
        html,
    )
    catalog: dict[str, str] = {}
    for url in matches:
        slug = url.rsplit("/", 1)[-1].split("-", 1)[1]
        catalog[slug] = url
    return catalog


def lookup_catalog_episode_url(title: str, show_id: str) -> str:
    title_slug = slugify(title)
    catalog = fetch_buzzsprout_episode_catalog(show_id)
    if not catalog:
        return ""
    if title_slug in catalog:
        return catalog[title_slug]

    title_key = normalize_key(title)
    best_url = ""
    best_score = 0
    for slug, url in catalog.items():
        slug_key = normalize_key(slug.replace("-", " "))
        shared = len(set(title_key.split()) & set(slug_key.split()))
        if title_slug.startswith(slug) or slug.startswith(title_slug):
            shared += 5
        if shared > best_score:
            best_score = shared
            best_url = url
    return best_url if best_score >= 4 else ""


def resolve_episode_url(channel_link: str, link: str, guid: str, title: str, rss_url: str = DEFAULT_RSS_URL) -> str:
    candidates = [normalize_space(link), normalize_space(guid)]
    for candidate in candidates:
        if candidate.startswith("http") and "/episodes/" in candidate:
            return candidate
    for candidate in candidates:
        if candidate.startswith("http") and "buzzsprout.com" in candidate:
            return candidate

    show_id = extract_show_id(channel_link, rss_url)
    episode_id = extract_episode_id(link, guid)
    if show_id and episode_id:
        return f"https://hoosierhealthmatters.buzzsprout.com/{show_id}/episodes/{episode_id}-{slugify(title)}"
    if show_id:
        return lookup_catalog_episode_url(title, show_id)
    return ""


def ensure_episode_url(episode: dict[str, Any], rss_url: str = DEFAULT_RSS_URL) -> str:
    existing = normalize_space(episode.get("episode_url") or "")
    if existing:
        return existing
    return resolve_episode_url(
        episode.get("channel_link", ""),
        episode.get("link", ""),
        episode.get("episode_id", ""),
        episode.get("title", ""),
        rss_url,
    )


class SearchEngine:
    def __init__(self, rss_url: str = DEFAULT_RSS_URL) -> None:
        self.rss_url = rss_url
        self.client = OpenAI() if OPENAI_CONFIGURED else None
        self.index_manifest: dict[str, Any] = {}
        self.episodes: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self.refresh(force=False)

    def refresh(self, force: bool = False) -> None:
        cache = self._load_cache()
        try:
            feed_episodes = self._fetch_feed_episodes()
            needs_full_rebuild = force or self._cache_needs_full_rebuild(cache)

            if needs_full_rebuild:
                indexed_episodes = [self._index_episode(episode) for episode in feed_episodes]
            else:
                cached_by_id = {episode["episode_id"]: episode for episode in cache["episodes"]}
                indexed_episodes = []
                for episode in feed_episodes:
                    cached = cached_by_id.get(episode["episode_id"])
                    if self._episode_needs_refresh(episode, cached):
                        indexed_episodes.append(self._index_episode(episode))
                    else:
                        cached["episode_url"] = ensure_episode_url(cached, self.rss_url)
                        indexed_episodes.append(cached)

            self.episodes = indexed_episodes
            self.chunks = [chunk for episode in self.episodes for chunk in episode.get("chunks", [])]
            self.index_manifest = {
                "cache_version": CACHE_VERSION,
                "rss_url": self.rss_url,
                "last_indexed_at": utc_now_iso(),
                "episode_count": len(self.episodes),
                "openai_enabled": OPENAI_CONFIGURED,
            }
            self._save_cache()
            self.last_error = None
        except Exception as exc:
            self.last_error = f"Archive refresh failed: {exc}"
            cached_episodes = cache.get("episodes", [])
            for episode in cached_episodes:
                episode["episode_url"] = ensure_episode_url(episode, self.rss_url)
            self.episodes = cached_episodes
            self.chunks = [chunk for episode in self.episodes for chunk in episode.get("chunks", [])]
            self.index_manifest = cache.get(
                "manifest",
                {
                    "cache_version": CACHE_VERSION,
                    "rss_url": self.rss_url,
                    "last_indexed_at": None,
                    "episode_count": len(self.episodes),
                    "openai_enabled": OPENAI_CONFIGURED,
                },
            )

    def search(self, query: str) -> dict[str, Any]:
        normalized_query = normalize_space(query)
        intent = self._interpret_query(normalized_query)
        query_embedding = self._embed_text(normalized_query) if self.client else []

        matched_episodes = self._retrieve_episodes(normalized_query, intent, query_embedding)
        top_episodes = matched_episodes[:4]
        supporting_chunks = self._retrieve_chunks(normalized_query, top_episodes, query_embedding)
        answer = self._answer_query(normalized_query, intent, top_episodes, supporting_chunks)
        support_note = None
        if not self.client:
            support_note = (
                "OpenAI is not configured, so search falls back to structured and keyword matching only."
            )
        return {
            "matched_episodes": top_episodes,
            "answer": answer,
            "support_note": support_note,
        }

    def _load_cache(self) -> dict[str, Any]:
        if not CACHE_PATH.exists():
            return {}
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_cache(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"manifest": self.index_manifest, "episodes": self.episodes}
        CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _cache_needs_full_rebuild(self, cache: dict[str, Any]) -> bool:
        if not cache:
            return True
        manifest = cache.get("manifest", {})
        if manifest.get("cache_version") != CACHE_VERSION:
            return True
        if manifest.get("rss_url") != self.rss_url:
            return True
        episodes = cache.get("episodes", [])
        if not episodes:
            return True
        if not any(ensure_episode_url(episode, self.rss_url) for episode in episodes):
            return True
        for episode in episodes:
            required = ["episode_id", "title", "published", "chunks"]
            if any(not episode.get(key) for key in required):
                return True
            for chunk in episode.get("chunks", []):
                if not chunk.get("chunk_text"):
                    return True
                if self.client and not chunk.get("embedding"):
                    return True
        return False

    def _episode_needs_refresh(self, fresh: dict[str, Any], cached: dict[str, Any] | None) -> bool:
        if not cached:
            return True
        tracked_fields = ["title", "published", "episode_url", "audio_url", "season", "episode_number"]
        for field in tracked_fields:
            if fresh.get(field) != cached.get(field):
                return True
        if not cached.get("transcript_text"):
            return True
        if not cached.get("chunks"):
            return True
        if self.client and any(not chunk.get("embedding") for chunk in cached["chunks"]):
            return True
        return False

    def _fetch_feed_episodes(self) -> list[dict[str, Any]]:
        rss_xml = fetch_bytes(self.rss_url, timeout=30)
        root = ET.fromstring(rss_xml)
        channel = root.find("channel")
        if channel is None:
            return []

        channel_link = normalize_space(channel.findtext("link", default="https://hoosierhealthmatters.buzzsprout.com/2446815"))
        ns = {
            "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
            "content": "http://purl.org/rss/1.0/modules/content/",
            "podcast": "https://podcastindex.org/namespace/1.0",
            "buzzsprout": "https://www.buzzsprout.com",
        }

        episodes: list[dict[str, Any]] = []
        for item in channel.findall("item"):
            title = normalize_space(item.findtext("title", default="Untitled episode"))
            description = strip_html(item.findtext("description", default=""))
            content_html = item.findtext("content:encoded", default="", namespaces=ns)
            content_text = strip_html(content_html)
            subtitle = strip_html(item.findtext("itunes:subtitle", default="", namespaces=ns))
            transcript_text = self._extract_transcript_from_item(item, ns)
            published, published_iso = parse_pub_date(item.findtext("pubDate", default=""))
            season, episode_number = extract_episode_numbers(title, subtitle, " ".join([description, content_text]))
            guid = normalize_space(item.findtext("guid", default="")) or normalize_space(
                item.findtext("link", default="")
            )
            link = normalize_space(item.findtext("link", default=""))
            enclosure = item.find("enclosure")
            audio_url = enclosure.attrib.get("url", "").strip() if enclosure is not None else ""
            episode_id = guid or f"{published}-{normalize_key(title)}"
            episode_url = resolve_episode_url(channel_link, link, guid, title, self.rss_url)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "title": title,
                    "published": published,
                    "published_iso": published_iso,
                    "season": season,
                    "episode_number": episode_number,
                    "episode_url": episode_url,
                    "audio_url": audio_url,
                    "description": description,
                    "summary_text": normalize_space(" ".join(part for part in [subtitle, description, content_text] if part)),
                    "transcript_text": transcript_text,
                    "channel_link": channel_link,
                    "link": link,
                }
            )
        episodes.sort(key=lambda episode: episode.get("published") or "", reverse=True)
        return episodes

    def _extract_transcript_from_item(self, item: ET.Element, ns: dict[str, str]) -> str:
        transcript_candidates: list[str] = []
        for tag in ["podcast:transcript", "transcript", "itunes:summary", "content:encoded", "description"]:
            text = item.findtext(tag, default="", namespaces=ns)
            if text:
                transcript_candidates.append(strip_html(text))
        for candidate in transcript_candidates:
            if looks_like_transcript(candidate):
                return candidate
        return ""

    def _index_episode(self, episode: dict[str, Any]) -> dict[str, Any]:
        transcript_text = episode.get("transcript_text") or ""
        transcript_source = "rss" if transcript_text else "missing"
        if not transcript_text and self.client and episode.get("audio_url"):
            transcript_text = self._transcribe_audio(episode["audio_url"])
            transcript_source = "transcribed" if transcript_text else "missing"

        if not transcript_text:
            transcript_text = episode.get("summary_text") or episode.get("description") or ""

        chunks = []
        embeddings: list[list[float]] = []
        for chunk_text in chunk_transcript(transcript_text):
            chunk_embedding = self._embed_text(chunk_text) if self.client else []
            chunks.append(
                {
                    "episode_id": episode["episode_id"],
                    "title": episode["title"],
                    "published": episode.get("published"),
                    "season": episode.get("season"),
                    "episode_number": episode.get("episode_number"),
                    "episode_url": episode.get("episode_url"),
                    "chunk_text": chunk_text,
                    "embedding": chunk_embedding,
                }
            )
            if chunk_embedding:
                embeddings.append(chunk_embedding)

        episode["episode_url"] = ensure_episode_url(episode, self.rss_url)
        episode["transcript_text"] = transcript_text
        episode["transcript_source"] = transcript_source
        episode["chunks"] = chunks
        episode["episode_embedding"] = mean_embedding(embeddings) if embeddings else []
        return episode

    def _download_audio_file(self, audio_url: str) -> Path | None:
        suffix = Path(audio_url).suffix or ".mp3"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)
        temp_file.close()
        try:
            temp_path.write_bytes(fetch_bytes(audio_url, timeout=120))
            return temp_path
        except Exception:
            temp_path.unlink(missing_ok=True)
            return None

    def _transcribe_audio(self, audio_url: str) -> str:
        if not self.client:
            return ""
        audio_path = self._download_audio_file(audio_url)
        if audio_path is None:
            return ""
        try:
            with audio_path.open("rb") as audio_handle:
                transcript = self.client.audio.transcriptions.create(
                    model=TRANSCRIPTION_MODEL,
                    file=audio_handle,
                )
            return normalize_space(getattr(transcript, "text", "") or "")
        except Exception:
            return ""
        finally:
            audio_path.unlink(missing_ok=True)

    def _embed_text(self, text: str) -> list[float]:
        if not self.client or not text:
            return []
        try:
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            return list(response.data[0].embedding)
        except Exception:
            return []

    def _interpret_query(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        season_match = re.search(r"season\s+(\d+)", lowered)
        episode_match = re.search(r"(episode|ep)\s+(\d+)", lowered)
        person_name = None
        patterns = [
            r"(?:interview|interviewed)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            r"(?:with|featuring)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            r"was\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+interviewed",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                person_name = match.group(1)
                break
        if person_name is None:
            title_case_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query)
            if title_case_names:
                person_name = title_case_names[0]

        intent = "topic_search"
        if season_match or episode_match:
            intent = "episode_lookup"
        elif any(term in lowered for term in ["interview", "interviewed", "guest", "who was", "appeared"]):
            intent = "guest_lookup"

        return {
            "intent": intent,
            "season": int(season_match.group(1)) if season_match else None,
            "episode_number": int(episode_match.group(2)) if episode_match else None,
            "person_name": person_name,
            "keywords": expand_keywords(query),
        }

    def _retrieve_episodes(
        self,
        query: str,
        intent: dict[str, Any],
        query_embedding: list[float],
    ) -> list[dict[str, Any]]:
        exact_matches: list[dict[str, Any]] = []
        if intent.get("season") or intent.get("episode_number"):
            for episode in self.episodes:
                if intent.get("season") and episode.get("season") != intent["season"]:
                    continue
                if intent.get("episode_number") and episode.get("episode_number") != intent["episode_number"]:
                    continue
                episode_copy = dict(episode)
                episode_copy["episode_url"] = ensure_episode_url(episode_copy, self.rss_url)
                episode_copy["score"] = 10.0
                episode_copy["match_reason"] = "Exact season and episode match"
                exact_matches.append(episode_copy)
        if exact_matches:
            return exact_matches

        query_terms = intent.get("keywords") or expand_keywords(query)
        person_name = normalize_key(intent.get("person_name") or "")
        scored: list[dict[str, Any]] = []
        for episode in self.episodes:
            summary_blob = normalize_key(
                " ".join([episode.get("title", ""), episode.get("summary_text", ""), episode.get("transcript_text", "")[:5000]])
            )
            title_blob = normalize_key(episode.get("title", ""))
            raw_summary = " ".join(
                [episode.get("title", ""), episode.get("summary_text", ""), episode.get("transcript_text", "")[:5000]]
            )
            score = 0.0
            reasons: list[str] = []

            if person_name and person_name in summary_blob:
                score += 5.0
                reasons.append("Guest name found in episode archive")
                interview_pattern = rf"(interview|guest|discussion).{{0,120}}{re.escape(intent.get('person_name', ''))}|{re.escape(intent.get('person_name', ''))}.{{0,120}}(interview|guest|discussion)"
                if re.search(interview_pattern, raw_summary, re.IGNORECASE):
                    score += 5.0
                    reasons = ["Direct guest interview match"]
                if person_name in title_blob:
                    score += 2.5
            elif intent["intent"] == "guest_lookup" and person_name:
                overlap = sum(part in summary_blob for part in person_name.split())
                score += overlap * 0.8

            keyword_hits = sum(term in summary_blob for term in query_terms)
            title_hits = sum(term in title_blob for term in query_terms)
            score += keyword_hits * 0.3 + title_hits * 0.5

            if query_embedding and episode.get("episode_embedding"):
                score += cosine_similarity(query_embedding, episode["episode_embedding"]) * 2.8

            if score > 0:
                episode_copy = dict(episode)
                episode_copy["episode_url"] = ensure_episode_url(episode_copy, self.rss_url)
                episode_copy["score"] = score
                episode_copy["match_reason"] = reasons[0] if reasons else "Relevant topic match"
                scored.append(episode_copy)

        scored.sort(key=lambda episode: (episode["score"], episode.get("published") or ""), reverse=True)
        return scored

    def _retrieve_chunks(
        self,
        query: str,
        episodes: list[dict[str, Any]],
        query_embedding: list[float],
    ) -> list[dict[str, Any]]:
        if not episodes:
            return []
        episode_ids = {episode["episode_id"] for episode in episodes}
        query_terms = expand_keywords(query)
        candidates: list[dict[str, Any]] = []
        for chunk in self.chunks:
            if chunk["episode_id"] not in episode_ids:
                continue
            chunk_blob = normalize_key(chunk["chunk_text"])
            keyword_score = sum(term in chunk_blob for term in query_terms) * 0.35
            semantic_score = cosine_similarity(query_embedding, chunk["embedding"]) * 2.2 if query_embedding else 0.0
            total_score = keyword_score + semantic_score
            if total_score > 0:
                chunk_copy = dict(chunk)
                chunk_copy["score"] = total_score
                candidates.append(chunk_copy)
        candidates.sort(key=lambda chunk: chunk["score"], reverse=True)
        return candidates[:6]

    def _answer_query(
        self,
        query: str,
        intent: dict[str, Any],
        episodes: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> str:
        if not episodes:
            return (
                "I couldn't find a supported answer in the indexed archive. Try a different guest name, topic, or season and episode reference."
            )

        if not self.client:
            return self._fallback_answer(query, intent, episodes, chunks)
        if intent["intent"] == "topic_search" and not chunks:
            return (
                "I found possibly relevant episodes, but I don't have enough retrieved excerpt text to answer that topic safely."
            )

        context_lines = []
        for episode in episodes:
            context_lines.append(
                f"EPISODE: {format_episode_label(episode)} | Published: {episode.get('published') or 'Unknown'}"
            )
        for index, chunk in enumerate(chunks, start=1):
            context_lines.append(f"EXCERPT {index}: {chunk['chunk_text']}")

        prompt = (
            "Answer the user using only the supplied episode metadata and excerpts. "
            "Be concise, trustworthy, and explicit when the archive does not fully support an answer. "
            "Mention episode title, season, episode number, and published date when relevant. "
            "Do not fabricate episode details or guest appearances."
        )
        try:
            response = self.client.responses.create(
                model=ANSWER_MODEL,
                input=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"Query: {query}"},
                            {"type": "input_text", "text": "\n".join(context_lines)},
                        ],
                    },
                ],
            )
            return normalize_space(response.output_text)
        except Exception:
            return self._fallback_answer(query, intent, episodes, chunks)

    def _fallback_answer(
        self,
        query: str,
        intent: dict[str, Any],
        episodes: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> str:
        lead = episodes[0]
        label = format_episode_label(lead)
        if intent["intent"] == "episode_lookup":
            published = lead.get("published") or "an unknown date"
            return f"The best exact match is {label}, published on {published}."
        if intent["intent"] == "guest_lookup":
            published = lead.get("published") or "an unknown date"
            return f"The archive most strongly points to {label}, published on {published}."
        if chunks:
            summary = chunks[0]["chunk_text"]
            return f"{label} appears most relevant. Based on the retrieved excerpt: {summary}"
        return f"{label} appears to be the closest archive match for '{query}'."