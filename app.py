from html import escape

import streamlit as st

from episode_index import (
    DEFAULT_RSS_URL,
    OPENAI_CONFIGURED,
    SearchEngine,
    format_episode_label,
    get_logo_path,
)


st.set_page_config(
    page_title="Hoosier Health Matters Episode Search",
    page_icon=str(get_logo_path()),
    layout="centered",
)


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: rgba(255, 255, 255, 0.84);
            --surface-strong: rgba(255, 255, 255, 0.96);
            --border: rgba(15, 23, 42, 0.08);
            --text: #0f172a;
            --muted: #475569;
            --accent: #0f766e;
            --accent-soft: rgba(15, 118, 110, 0.12);
            --shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --surface: rgba(15, 23, 42, 0.72);
                --surface-strong: rgba(15, 23, 42, 0.88);
                --border: rgba(148, 163, 184, 0.2);
                --text: #e5eef8;
                --muted: #c5d3e1;
                --accent: #5eead4;
                --accent-soft: rgba(94, 234, 212, 0.12);
                --shadow: 0 18px 45px rgba(2, 6, 23, 0.45);
            }
        }

        .stApp {
            background:
                radial-gradient(circle at top, rgba(20, 184, 166, 0.16), transparent 32%),
                linear-gradient(180deg, rgba(248, 250, 252, 0.94), rgba(241, 245, 249, 0.98));
        }

        @media (prefers-color-scheme: dark) {
            .stApp {
                background:
                    radial-gradient(circle at top, rgba(45, 212, 191, 0.14), transparent 34%),
                    linear-gradient(180deg, rgba(2, 6, 23, 0.98), rgba(15, 23, 42, 1));
            }
        }

        .main .block-container {
            max-width: 860px;
            padding-top: 2.6rem;
            padding-bottom: 3rem;
        }

        .hero-shell {
            text-align: center;
            margin-bottom: 1.4rem;
        }

        .hero-logo {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }

        .hero-title {
            color: var(--text);
            font-size: clamp(2rem, 3vw, 2.8rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.45rem;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 680px;
            margin: 0 auto;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 24px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(14px);
            padding: 1.35rem 1.35rem 1.15rem 1.35rem;
            margin-bottom: 1rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
            letter-spacing: -0.02em;
        }

        .episode-card {
            background: var(--surface-strong);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        }

        .episode-card-title {
            color: var(--text);
            font-size: 1.03rem;
            font-weight: 800;
            line-height: 1.5;
            margin-bottom: 0.3rem;
        }

        .episode-meta {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }

        .episode-excerpt {
            color: var(--text);
            font-size: 0.97rem;
            line-height: 1.65;
            margin-top: 0.7rem;
        }

        .episode-excerpt-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .episode-stamp {
            display: inline-block;
            color: var(--accent);
            background: var(--accent-soft);
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin: 0.55rem 0 0.15rem 0;
        }

        .result-section-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin: 1.25rem 0 0.7rem 0;
            letter-spacing: -0.02em;
        }

        .answer-copy {
            color: var(--text);
            font-size: 1.02rem;
            line-height: 1.78;
        }

        .helper-note {
            color: var(--muted);
            text-align: center;
            font-size: 0.93rem;
            margin-top: 0.35rem;
        }

        .status-chip {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.86rem;
            font-weight: 700;
            margin-top: 0.4rem;
        }

        div[data-testid="stTextInputRootElement"] > div {
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_engine() -> SearchEngine:
    return SearchEngine()


def render_episode_list(episodes: list[dict]) -> None:
    if not episodes:
        st.markdown('<div class="result-section-title">Matched episode(s)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><div class="answer-copy">No matching episodes were found in the indexed archive.</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="result-section-title">Matched episode(s)</div>', unsafe_allow_html=True)
    for episode in episodes:
        label = format_episode_label(episode)
        meta = []
        if episode.get("published"):
            meta.append(episode["published"])
        if episode.get("match_reason"):
            meta.append(episode["match_reason"])
        with st.container():
            safe_label = escape(label)
            st.markdown('<div class="episode-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="episode-card-title">{safe_label}</div>',
                unsafe_allow_html=True,
            )
            if meta:
                safe_meta = escape(" • ".join(meta))
                st.markdown(
                    f'<div class="episode-meta">{safe_meta}</div>',
                    unsafe_allow_html=True,
                )
            if episode.get("discussion_timestamp"):
                stamp_label = escape(episode["discussion_timestamp"])
                if episode.get("discussion_timestamp_approx"):
                    stamp_label = f"Around {stamp_label} in the episode"
                else:
                    stamp_label = f"At {stamp_label} in the episode"
                st.markdown(
                    f'<div class="episode-stamp">{stamp_label}</div>',
                    unsafe_allow_html=True,
                )
            if episode.get("discussion_excerpt"):
                safe_excerpt = escape(episode["discussion_excerpt"])
                st.markdown(
                    f'<div class="episode-excerpt"><div class="episode-excerpt-label">Discusses</div>{safe_excerpt}</div>',
                    unsafe_allow_html=True,
                )
            if episode.get("episode_url"):
                st.link_button("Open episode", episode["episode_url"])
            st.markdown("</div>", unsafe_allow_html=True)


def render_answer(answer: str) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-copy">{answer}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def is_transcript_safety_fallback(answer: str) -> bool:
    normalized = " ".join((answer or "").split()).lower()
    return normalized.startswith(
        "i found relevant episodes, but i don't have transcript-backed excerpts to answer that question safely yet"
    ) or normalized.startswith(
        "i found relevant episodes, but i don’t have transcript-backed excerpts to answer that question safely yet"
    )


def render_sidebar(engine: SearchEngine) -> None:
    with st.sidebar:
        st.markdown("### Archive status")
        manifest = engine.index_manifest
        if engine.last_error:
            st.error(engine.last_error)
        if manifest.get("last_indexed_at"):
            st.caption(f"Last refreshed: {manifest['last_indexed_at']}")
        st.caption(f"Episodes indexed: {len(engine.episodes)}")
        if manifest.get("rss_url"):
            st.caption(f"RSS feed: {manifest['rss_url']}")
        if OPENAI_CONFIGURED:
            st.markdown('<div class="status-chip">OpenAI enabled</div>', unsafe_allow_html=True)
        else:
            st.warning(
                "Set `OPENAI_API_KEY` to enable embeddings, answer generation, and transcript fallback for episodes without RSS transcripts."
            )
        if st.button("Refresh archive", use_container_width=True):
            engine.refresh(force=False)
            st.rerun()
        if st.button("Full rebuild", use_container_width=True):
            engine.refresh(force=True)
            st.rerun()
        with st.expander("Search examples"):
            st.markdown(
                "\n".join(
                    [
                        "- `When did we interview Veronica Vernon?`",
                        "- `In what episode was Jane Hartsock interviewed?`",
                        "- `What episode discussed Supreme Court rulings?`",
                        "- `What did we say about Medicaid reimbursement?`",
                        "- `Season 2 Episode 11`",
                    ]
                )
            )


def main() -> None:
    render_styles()
    engine = get_engine()
    render_sidebar(engine)

    st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
    st.image(str(get_logo_path()), width=108)
    st.markdown(
        '<div class="hero-title">Hoosier Health Matters Episode Search</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-subtitle">Search the podcast archive in plain language to find interviews, episode references, and trusted topic summaries grounded in real transcript excerpts.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Search the archive",
        placeholder="Ask about a guest, a topic, or a season and episode number…",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="helper-note">Examples: "When did we interview Veronica Vernon?" or "What did we say about Medicaid reimbursement?"</div>',
        unsafe_allow_html=True,
    )

    if query:
        with st.spinner("Searching the archive..."):
            result = engine.search(query)
        render_episode_list(result["matched_episodes"])
        if not is_transcript_safety_fallback(result["answer"]):
            render_answer(result["answer"])
        if result.get("support_note"):
            st.caption(result["support_note"])
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ready to search</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="answer-copy">The archive refreshes from the Buzzsprout RSS feed and only answers from retrieved episode excerpts. Use the sidebar to refresh when a new episode is published.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
