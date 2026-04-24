"""Build or refresh the Hoosier Health Matters search index.

This keeps archive ingestion separate from the public Streamlit app. Run it
manually, from an admin workflow, or from a scheduled job when you want to
refresh the prepared JSON index.
"""

from __future__ import annotations

import argparse

from episode_index import CACHE_PATH, SearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local episode search index.")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only reindex new or changed RSS entries when the existing cache is usable.",
    )
    args = parser.parse_args()

    engine = SearchEngine()
    if not args.incremental:
        engine.refresh(force=True)

    if engine.last_error:
        raise SystemExit(engine.last_error)

    print(f"Indexed {len(engine.episodes)} episodes.")
    print(f"Wrote {CACHE_PATH}")


if __name__ == "__main__":
    main()
