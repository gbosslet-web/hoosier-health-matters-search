from pathlib import Path

import streamlit as st

from episode_index import OPENAI_CONFIGURED, SearchEngine, format_episode_label, get_logo_path


st.set_page_config(
    page_title="Hoosier Health Matters Episode Search",
    page_icon="🎙️",
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
            padding: 0.95rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        }

        .episode-label {
            color: var(--text);
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }

        .episode-label a {
            color: var(--accent);
            text-decoration: none;
        }

        .episode-label a:hover {
            text-decoration: underline;
        }

        .episode-meta {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.25rem;
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


def render_logo() -> None:
    logo_path = get_logo_path()
    try:
        image_bytes = Path(logo_path).read_bytes()
        st.image(image_bytes, width=108)
    except Exception:
        st.markdown(
            '<div style="font-size:3rem; line-height:1; margin-bottom:0.35rem;">🎙️</div>',
            unsafe_allow_html=True,
        )


def render_episode_list(episodes: list[dict]) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Matched episode(s)</div>', unsafe_allow_html=True)
    if not episodes:
        st.markdown(
            '<div class="answer-copy">No matching episodes were found in the indexed archive.</div>',
            unsafe_allow_html=True,
        )
    for episode in episodes:
        label = format_episode_label(episode)
        meta = []
        if episode.get("published"):
            meta.append(episode["published"])
        if episode.get("match_reason"):
            meta.append(episode["match_reason"])
        st.markdown('<div class="episode-card">', unsafe_allow_html=True)
        if episode.get("episode_url"):
            safe_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe_url = episode["episode_url"].replace('"', "%22")
            st.markdown(
                f'<div class="episode-label"><a href="{safe_url}" target="_blank">{safe_label}</a></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="episode-label">{label}</div>', unsafe_allow_html=True)
        if meta:
            st.markdown(f'<div class="episode-meta">{" • ".join(meta)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_answer(answer: str) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-copy">{answer}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


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
    render_logo()
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
        placeholder="Ask about a guest, a topic, or a season and episode number...",
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
