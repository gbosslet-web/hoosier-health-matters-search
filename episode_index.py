from __future__ import annotations

import json
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from functools import lru_cache

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_PATH = DATA_DIR / "archive_index.json"
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_RSS_URL = os.getenv("HHM_RSS_URL", "https://feeds.buzzsprout.com/2446815.rss")
CACHE_VERSION = 7
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
ANSWER_MODEL = os.getenv("OPENAI_RESPONSE_MODEL", "gpt-4.1-mini")
TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
OPENAI_CONFIGURED = bool(os.getenv("OPENAI_API_KEY"))
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
SHOW_TITLE = "Hoosier Health Matters"
BUZZSPROUT_EPISODES_URL = os.getenv(
    "HHM_BUZZSPROUT_EPISODES_URL",
    "https://hoosierhealthmatters.buzzsprout.com/2446815/episodes",
)
APPLE_SHOW_URL = os.getenv(
    "HHM_APPLE_SHOW_URL",
    "https://podcasts.apple.com/us/podcast/hoosier-health-matters/id1793605849",
)


class PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self.parts)


class AppleEpisodeLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        if "/podcast/" in href and "?i=" in href:
            self._active_href = href
            self._active_parts = []

    def handle_data(self, data: str) -> None:
        if self._active_href and data.strip():
            self._active_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._active_href:
            return
        text = normalize_space(" ".join(self._active_parts))
        if text:
            self.links.append((self._active_href, text))
        self._active_href = None
        self._active_parts = []


class EpisodeLinkParser(HTMLParser):
    def __init__(self, href_predicate) -> None:
        super().__init__()
        self.href_predicate = href_predicate
        self.links: list[tuple[str, str]] = []
        self._active_href: str | None = None
        self._active_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        if self.href_predicate(href):
            self._active_href = href
            self._active_parts = []

    def handle_data(self, data: str) -> None:
        if self._active_href and data.strip():
            self._active_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._active_href:
            return
        text = normalize_space(" ".join(self._active_parts))
        if text:
            self.links.append((self._active_href, text))
        self._active_href = None
        self._active_parts = []


def get_logo_path() -> Path:
    candidates = [
        ASSETS_DIR / "HHM logo.jpeg",
        ASSETS_DIR / "logo.jpeg",
        ASSETS_DIR / "logo.jpg",
        ASSETS_DIR / "logo.svg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ASSETS_DIR / "logo.svg"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


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


def extract_person_name(query: str) -> str | None:
    patterns = [
        r"(?:interview|interviewed|feature|featured|featuring)\s+([A-Za-z][A-Za-z'.-]+(?:\s+[A-Za-z][A-Za-z'.-]+)+)",
        r"(?:with)\s+([A-Za-z][A-Za-z'.-]+(?:\s+[A-Za-z][A-Za-z'.-]+)+)",
        r"was\s+([A-Za-z][A-Za-z'.-]+(?:\s+[A-Za-z][A-Za-z'.-]+)+)\s+interviewed",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = normalize_space(match.group(1).strip(" ?!.,"))
            if len(candidate.split()) >= 2:
                return candidate

    title_case_names = re.findall(r"\b[A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+)+\b", query)
    if title_case_names:
        return normalize_space(title_case_names[0])
    return None


def guest_name_match_score(person_name: str, text: str) -> float:
    if not person_name or not text:
        return 0.0
    person_parts = [part for part in normalize_key(person_name).split() if part]
    text_tokens = [token for token in normalize_key(text).split() if token]
    if not person_parts or not text_tokens:
        return 0.0

    score = 0.0
    for part in person_parts:
        best = 0.0
        for token in text_tokens:
            if part == token:
                best = max(best, 1.0)
            else:
                best = max(best, SequenceMatcher(None, part, token).ratio())
        if best >= 0.9:
            score += 1.0
        elif best >= 0.8:
            score += 0.7
        elif best >= 0.7:
            score += 0.4
    return score


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


def safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


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


def parse_duration_seconds(value: str | None) -> int | None:
    if not value:
        return None
    raw = normalize_space(str(value))
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    parts = raw.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = [int(part) for part in parts]
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = [int(part) for part in parts]
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None
    return None


def format_timestamp(seconds: int | None) -> str | None:
    if seconds is None or seconds < 0:
        return None
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


TIMELINE_MARKER_RE = re.compile(r"(?P<ts>\b\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*")
LOW_SIGNAL_MARKERS = [
    "intro",
    "wrap up",
    "thanks for listening",
    "join the good trouble",
    "board member",
    "board members",
    "become a member",
    "donate to support",
    "follow us on bluesky",
    "follow us on instagram",
]


def strip_episode_boilerplate(text: str) -> str:
    normalized = normalize_space(text)
    if not normalized:
        return ""
    normalized = re.sub(
        r"Hoosier Health Matters\s+Season\s+\d+,\s*Episode\s+\d+\s+Date:\s*[^T]+Title:\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bTitle:\s*", "", normalized, flags=re.IGNORECASE)
    return normalized.strip(" -")


def split_excerpt_units(text: str) -> list[str]:
    normalized = strip_episode_boilerplate(text)
    if not normalized:
        return []
    units = re.split(r"(?:(?<=[.!?])\s+|\s+\d{1,2}:\d{2}\s*[-–]\s*)", normalized)
    return [unit.strip(" -") for unit in units if unit.strip(" -")]


def extract_timeline_segments(text: str) -> list[dict[str, str]]:
    normalized = strip_episode_boilerplate(text)
    if not normalized:
        return []
    matches = list(TIMELINE_MARKER_RE.finditer(normalized))
    if not matches:
        return []

    segments: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        segment_text = normalize_space(normalized[start:end].strip(" -"))
        if segment_text:
            segment_key = normalize_key(segment_text)
            if any(marker in segment_key for marker in LOW_SIGNAL_MARKERS if marker != "intro"):
                continue
            segments.append({"timestamp": match.group("ts"), "text": segment_text})
    return segments


def score_excerpt_candidate(text: str, query: str) -> float:
    text_key = normalize_key(text)
    keywords = expand_keywords(query)
    score = float(sum(term in text_key for term in keywords))
    query_key = normalize_key(query)
    if query_key and query_key in text_key:
        score += 2.5
    if any(marker in text_key for marker in LOW_SIGNAL_MARKERS):
        score -= 2.0
    return score


def clean_display_excerpt(text: str, max_chars: int = 200) -> str:
    cleaned = normalize_space(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\((?:here|see|link|newsletter)[^)]+\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:here we discuss|gabe and tracey discuss|discussion of)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^in this episode[^.]*\.\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = normalize_space(cleaned).strip(" -.,;:")
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def make_query_focused_excerpt(text: str, query: str, max_chars: int = 240) -> str:
    units = split_excerpt_units(text)
    if not units:
        return ""
    best_unit = units[0]
    best_score = -1.0
    for unit in units:
        score = score_excerpt_candidate(unit, query)
        if len(unit) <= max_chars:
            score += 0.2
        if score > best_score:
            best_score = score
            best_unit = unit
    if len(best_unit) <= max_chars:
        return best_unit
    truncated = best_unit[: max_chars - 1].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def select_support_snippet(text: str, query: str, max_chars: int = 240) -> dict[str, str | None]:
    segments = extract_timeline_segments(text)
    if segments:
        best_segment = max(segments, key=lambda segment: score_excerpt_candidate(segment["text"], query))
        best_score = score_excerpt_candidate(best_segment["text"], query)
        if best_score <= 0:
            return {"excerpt": None, "timestamp": None}
        excerpt = make_query_focused_excerpt(best_segment["text"], query, max_chars=max_chars)
        excerpt = clean_display_excerpt(excerpt, max_chars=max_chars)
        return {"excerpt": excerpt, "timestamp": best_segment["timestamp"]}

    excerpt = make_query_focused_excerpt(text, query, max_chars=max_chars)
    if excerpt and score_excerpt_candidate(excerpt, query) <= 0:
        return {"excerpt": None, "timestamp": None}
    return {"excerpt": clean_display_excerpt(excerpt, max_chars=max_chars), "timestamp": None}


def chunk_transcript(text: str, target_words: int = 180, overlap_words: int = 45) -> list[dict[str, Any]]:
    words = text.split()
    if len(words) <= target_words:
        return [{"chunk_text": normalize_space(text), "start_word": 0, "end_word": len(words)}]
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk = normalize_space(" ".join(words[start:end]))
        if chunk:
            chunks.append({"chunk_text": chunk, "start_word": start, "end_word": end})
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


def fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    return json.loads(fetch_bytes(url, timeout=timeout).decode("utf-8"))


def derive_buzzsprout_episode_url_from_audio(audio_url: str) -> str:
    if not audio_url.startswith("http"):
        return ""
    match = re.search(r"(https://www\.buzzsprout\.com/\d+/episodes/\d+-[^/?#]+)\.mp3(?:$|[?#])", audio_url)
    if not match:
        return ""
    return match.group(1)


def apple_title_key(value: str) -> str:
    return normalize_key(value).replace(" scotus ", " supreme court ")


def apple_match_score(left: str, right: str) -> int:
    left_key = apple_title_key(left)
    right_key = apple_title_key(right)
    if not left_key or not right_key:
        return 0
    if left_key == right_key:
        return 100
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    shared = len(left_tokens & right_tokens)
    if left_key in right_key or right_key in left_key:
        shared += 4
    return shared


@lru_cache(maxsize=256)
def lookup_apple_episode_url(show_title: str, episode_title: str) -> str:
    search_term = f"{show_title} {episode_title}"
    search_params = urllib.parse.urlencode(
        {
            "term": search_term,
            "media": "podcast",
            "entity": "podcastEpisode",
            "limit": 25,
        }
    )
    search_url = f"https://itunes.apple.com/search?{search_params}"
    try:
        search_payload = fetch_json(search_url, timeout=20)
    except Exception:
        return ""

    best_url = ""
    best_score = 0
    show_key = apple_title_key(show_title)
    for result in search_payload.get("results", []):
        track_title = normalize_space(result.get("trackName", ""))
        podcast_name = normalize_space(result.get("collectionName", ""))
        episode_url = result.get("episodeUrl") or result.get("trackViewUrl") or ""
        if not track_title or not episode_url.startswith("http"):
            continue
        show_score = apple_match_score(show_title, podcast_name)
        if show_score < 2 and show_key not in apple_title_key(podcast_name):
            continue
        title_score = apple_match_score(episode_title, track_title)
        total_score = title_score + show_score
        if total_score > best_score:
            best_score = total_score
            best_url = episode_url
    return best_url if best_score >= 6 else ""


@lru_cache(maxsize=1)
def fetch_apple_show_page_links() -> list[tuple[str, str]]:
    try:
        html = fetch_bytes(APPLE_SHOW_URL, timeout=20).decode("utf-8", errors="ignore")
    except Exception:
        return []

    parser = AppleEpisodeLinkParser()
    parser.feed(html)
    return parser.links


@lru_cache(maxsize=1)
def fetch_buzzsprout_episode_links() -> list[tuple[str, str]]:
    try:
        html = fetch_bytes(BUZZSPROUT_EPISODES_URL, timeout=20).decode("utf-8", errors="ignore")
    except Exception:
        return []

    parser = EpisodeLinkParser(lambda href: "/episodes/" in href)
    parser.feed(html)
    links: list[tuple[str, str]] = []
    for href, text in parser.links:
        absolute = urllib.parse.urljoin(BUZZSPROUT_EPISODES_URL, href)
        links.append((absolute, text))
    return links


def lookup_apple_show_page_episode_url(episode_title: str) -> str:
    best_url = ""
    best_score = 0
    for url, link_text in fetch_apple_show_page_links():
        score = apple_match_score(episode_title, link_text)
        if score > best_score:
            best_score = score
            best_url = url
    return best_url if best_score >= 6 else ""


def lookup_buzzsprout_episode_url(episode_title: str) -> str:
    best_url = ""
    best_score = 0
    for url, link_text in fetch_buzzsprout_episode_links():
        score = apple_match_score(episode_title, link_text)
        if score > best_score:
            best_score = score
            best_url = url
    return best_url if best_score >= 6 else ""


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
        top_episodes = self._attach_episode_support(top_episodes, supporting_chunks, normalized_query)
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

    def _attach_episode_support(
        self,
        episodes: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        best_chunk_by_episode: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            episode_id = chunk["episode_id"]
            existing = best_chunk_by_episode.get(episode_id)
            if existing is None or chunk.get("score", 0) > existing.get("score", 0):
                best_chunk_by_episode[episode_id] = chunk

        enriched: list[dict[str, Any]] = []
        for episode in episodes:
            episode_copy = dict(episode)
            best_chunk = best_chunk_by_episode.get(episode["episode_id"])
            if best_chunk:
                source_text = best_chunk["chunk_text"]
                if episode_copy.get("transcript_source") in {"rss", "missing"}:
                    source_text = episode_copy.get("summary_text") or source_text
                support = select_support_snippet(source_text, query)
                episode_copy["discussion_excerpt"] = support.get("excerpt")
                if support.get("timestamp"):
                    episode_copy["discussion_timestamp"] = support["timestamp"]
                    episode_copy["discussion_timestamp_approx"] = False
            enriched.append(episode_copy)
        return enriched

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
        if not any(episode.get("episode_url") for episode in episodes):
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
            duration_seconds = parse_duration_seconds(item.findtext("itunes:duration", default="", namespaces=ns))
            season, episode_number = extract_episode_numbers(title, subtitle, " ".join([description, content_text]))
            guid = normalize_space(item.findtext("guid", default="")) or normalize_space(
                item.findtext("link", default="")
            )
            link = normalize_space(item.findtext("link", default=""))
            enclosure = item.find("enclosure")
            audio_url = enclosure.attrib.get("url", "").strip() if enclosure is not None else ""
            episode_id = guid or f"{published}-{normalize_key(title)}"
            derived_episode_url = derive_buzzsprout_episode_url_from_audio(audio_url)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "title": title,
                    "published": published,
                    "published_iso": published_iso,
                    "season": season,
                    "episode_number": episode_number,
                    "episode_url": self._resolve_episode_url(title, link, audio_url, SHOW_TITLE) or derived_episode_url,
                    "audio_url": audio_url,
                    "duration_seconds": duration_seconds,
                    "description": description,
                    "summary_text": normalize_space(" ".join(part for part in [subtitle, description, content_text] if part)),
                    "transcript_text": transcript_text,
                }
            )
        episodes.sort(key=lambda episode: episode.get("published") or "", reverse=True)
        return episodes

    def _resolve_episode_url(self, title: str, rss_link: str, audio_url: str, show_title: str) -> str:
        if rss_link.startswith("http"):
            return rss_link
        derived_audio_url = derive_buzzsprout_episode_url_from_audio(audio_url)
        if derived_audio_url:
            return derived_audio_url
        buzzsprout_url = lookup_buzzsprout_episode_url(title)
        if buzzsprout_url:
            return buzzsprout_url
        apple_url = lookup_apple_episode_url(show_title, title)
        if apple_url:
            return apple_url
        return lookup_apple_show_page_episode_url(title)

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
        transcript_words = max(len(transcript_text.split()), 1)
        duration_seconds = episode.get("duration_seconds")
        for chunk_info in chunk_transcript(transcript_text):
            chunk_text = chunk_info["chunk_text"]
            start_word = chunk_info.get("start_word", 0)
            start_seconds = None
            if duration_seconds:
                start_seconds = int(duration_seconds * (start_word / transcript_words))
            chunk_embedding = self._embed_text(chunk_text) if self.client else []
            chunks.append(
                {
                    "episode_id": episode["episode_id"],
                    "title": episode["title"],
                    "published": episode.get("published"),
                    "season": episode.get("season"),
                    "episode_number": episode.get("episode_number"),
                    "episode_url": episode.get("episode_url"),
                    "timestamp_seconds": start_seconds,
                    "timestamp_label": format_timestamp(start_seconds),
                    "chunk_text": chunk_text,
                    "embedding": chunk_embedding,
                }
            )
            if chunk_embedding:
                embeddings.append(chunk_embedding)

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
        person_name = extract_person_name(query)

        intent = "topic_search"
        if season_match or episode_match:
            intent = "episode_lookup"
        elif any(term in lowered for term in ["interview", "interviewed", "guest", "who was", "appeared", "feature", "featured", "featuring"]):
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
                episode_copy["score"] = 10.0
                episode_copy["match_reason"] = "Exact season and episode match"
                exact_matches.append(episode_copy)
        if exact_matches:
            return exact_matches

        query_terms = intent.get("keywords") or expand_keywords(query)
        person_name = intent.get("person_name") or ""
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

            if person_name and guest_name_match_score(person_name, raw_summary) >= 1.7:
                score += 5.0
                reasons.append("Guest name found in episode archive")
                person_pattern = re.escape(intent.get("person_name", ""))
                interview_pattern = rf"(interview|guest|discussion|featured|featuring).{{0,120}}{person_pattern}|{person_pattern}.{{0,120}}(interview|guest|discussion|featured|featuring)"
                fuzzy_interview = guest_name_match_score(person_name, raw_summary) >= 1.7 and any(
                    term in raw_summary.lower() for term in ["interview", "guest", "featured", "featuring"]
                )
                if re.search(interview_pattern, raw_summary, re.IGNORECASE) or fuzzy_interview:
                    score += 5.0
                    reasons = ["Direct guest interview match"]
                if guest_name_match_score(person_name, episode.get("title", "")) >= 1.0:
                    score += 2.5
            elif intent["intent"] == "guest_lookup" and person_name:
                score += guest_name_match_score(person_name, raw_summary) * 0.8

            keyword_hits = sum(term in summary_blob for term in query_terms)
            title_hits = sum(term in title_blob for term in query_terms)
            score += keyword_hits * 0.3 + title_hits * 0.5

            if query_embedding and episode.get("episode_embedding"):
                score += cosine_similarity(query_embedding, episode["episode_embedding"]) * 2.8

            if score > 0:
                episode_copy = dict(episode)
                episode_copy["score"] = score
                episode_copy["match_reason"] = reasons[0] if reasons else "Relevant topic match"
                scored.append(episode_copy)

        scored.sort(key=lambda episode: (episode["score"], episode.get("published") or ""), reverse=True)
        if intent["intent"] == "guest_lookup":
            direct_matches = [episode for episode in scored if episode.get("match_reason") == "Direct guest interview match"]
            if direct_matches:
                return direct_matches
            named_matches = [episode for episode in scored if episode.get("match_reason") == "Guest name found in episode archive"]
            if named_matches:
                return named_matches
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
            semantic_score = cosine_similarity(query_embedding, chunk[