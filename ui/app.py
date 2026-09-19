"""Run from the repository root: python -m streamlit run ui/app.py."""

import asyncio
from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

# Streamlit executes this file as a script; make the existing packages importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import create_assistant
from src.exceptions import AllSourcesFailedError, InvalidQuestionError
from src.models import MAX_QUESTION_LENGTH, SourceName


async def research(question: str, sources: list[SourceName]):
    """Keep each assistant and its async resources within one request's loop."""
    assistant = create_assistant()
    result = await assistant.ask(question, sources=sources)
    return result, assistant.last_degradation_notes


def main() -> None:
    st.set_page_config(page_title="Async Research Assistant")
    st.title("Async Research Assistant")
    st.caption("Research Wikipedia, arXiv, and the web concurrently.")

    source_options = {
        "Wikipedia": SourceName.WIKI,
        "arXiv": SourceName.ARXIV,
        "Web": SourceName.WEB,
    }
    with st.form("research"):
        question = st.text_area("Research question", max_chars=MAX_QUESTION_LENGTH)
        selected = st.multiselect(
            "Sources", options=list(source_options), default=list(source_options)
        )
        submitted = st.form_submit_button("Research")

    if not submitted:
        return
    if not question.strip():
        st.error("Please enter a research question.")
        return
    if not selected:
        st.error("Please select at least one source.")
        return

    try:
        # The existing provider adapters read os.getenv(), unlike Settings.
        load_dotenv(ROOT / ".env", override=False)
        with st.spinner("Researching..."):
            result, notes = asyncio.run(
                research(question, [source_options[label] for label in selected])
            )
    except (InvalidQuestionError, AllSourcesFailedError) as error:
        st.error(str(error))
        return
    except Exception:
        st.error(
            "Research could not be completed. Check your provider settings, "
            "API keys, and network connection, then try again."
        )
        return

    st.markdown(result.answer)
    st.subheader("Sources")
    for citation in sorted(result.citations, key=lambda item: item.index):
        source = citation.source
        st.text(f"[{citation.index}] ({source.origin}) {source.title}")
        st.write(source.url)
    if notes:
        unavailable = ", ".join(note.source.value for note in notes)
        st.warning(
            f"Some sources were unavailable ({unavailable}). "
            "The answer uses the remaining sources."
        )


if __name__ == "__main__":
    main()
