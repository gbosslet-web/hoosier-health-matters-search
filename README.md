# Hoosier Health Matters Episode Search

A clean Streamlit search app for the Hoosier Health Matters podcast archive. It indexes the Buzzsprout RSS feed, uses OpenAI embeddings for retrieval, and answers only from retrieved archive excerpts.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

4. Run the app:

```bash
streamlit run app.py
```

## How the archive cache works

- The archive is stored locally at `data/archive_index.json`.
- On startup, the app fetches the RSS feed and compares feed episodes with the local cache.
- If the cache is missing, outdated, or incomplete, the app rebuilds the index.
- If only new or changed RSS entries appear, it reindexes just those episodes and keeps the rest.
- The sidebar includes `Refresh archive` for incremental updates and `Full rebuild` for a clean refresh.

## Transcript fallback behavior

- The app first looks for transcript-like text in RSS fields such as transcript, summary, HTML content, and description fields.
- If a transcript is missing and `OPENAI_API_KEY` is available, it downloads the episode audio and sends it to OpenAI transcription.
- The resulting transcript is chunked, embedded, and stored locally so the next search is fast.
- If no transcript is available and transcription is not possible, the app falls back to indexing the episode summary text, but search quality will be lower.

## Retrieval design

- Query interpretation detects likely episode lookup, guest lookup, or broad topic search.
- Retrieval prefers exact season and episode matches first.
- Guest-oriented searches then look for person-name evidence in episode text.
- Broad topic searches use episode-level semantic ranking first, then chunk-level ranking only within top episodes.
- Final answers are generated from the retrieved excerpts only.

## Environment variables

- `OPENAI_API_KEY`: required for embeddings, final answers, and transcript fallback.
- `HHM_RSS_URL`: optional override for the podcast RSS feed.
- `OPENAI_EMBEDDING_MODEL`: optional embedding model override.
- `OPENAI_RESPONSE_MODEL`: optional Responses API model override.
- `OPENAI_TRANSCRIPTION_MODEL`: optional transcription model override.
