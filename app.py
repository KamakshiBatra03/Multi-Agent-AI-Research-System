import streamlit as st
from datetime import datetime

from agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent AI Research System",
    page_icon="🔎",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_text(content) -> str:
    """Normalize LLM output (str, message object, or list of content blocks) to a string."""
    if content is None:
        return ""
    if hasattr(content, "content"):  # LangChain message object
        content = content.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def run_pipeline(topic: str, status_box) -> dict:
    """Same logic as callerFunc.caller_func, but reports progress to the UI."""
    state = {}

    # Step 1 - Search agent
    status_box.update(label="Step 1/4 · Search agent is working…", state="running")
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"""
            Search for recent and reliable information about {topic}.

            IMPORTANT:
            Return at least 5 search results.

            For EACH result provide:
            1. Title
            2. Actual URL (complete https:// link)
            3. Snippet

            Do not provide only a summary.
            Do not omit URLs.
            """)]
    })
    state["search_results"] = to_text(search_result["messages"][-1].content)
    st.write("✅ Search agent finished")

    # Step 2 - Reader agent
    status_box.update(label="Step 2/4 · Reader agent is scraping top resources…", state="running")
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({"messages": [("user", f"""Based on the following search results about {topic},
         pick the most relevant URL and scrape it for deeper content.
         Search results:
         {state['search_results'][:800]}
        Instructions:
            1. Extract the actual URLs from the search results.
            2. Select the most relevant and reliable URL.
            3. Call the web_scraper tool using that URL.
            4. Return the scraped content.
            5. Do not ask the user for a URL.""")]
    })
    state["scraped_result"] = to_text(reader_result["messages"][-1].content)
    st.write("✅ Reader agent finished")

    # Step 3 - Writer chain
    status_box.update(label="Step 3/4 · Writer is drafting the report…", state="running")
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED RESULTS:\n{state['scraped_result']}"
    )
    state["report"] = to_text(writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    }))
    st.write("✅ Writer finished")

    # Step 4 - Critic chain
    status_box.update(label="Step 4/4 · Critic is reviewing the report…", state="running")
    state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
    st.write("✅ Critic finished")

    return state


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of (topic, timestamp, result)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Pipeline")
    st.markdown(
        """
        1. **Search Agent** – finds recent, reliable sources  
        2. **Reader Agent** – scrapes the best URL for depth  
        3. **Writer Chain** – drafts the research report  
        4. **Critic Chain** – reviews and gives feedback
        """
    )
    st.divider()

    st.subheader("🕘 History")
    if st.session_state.history:
        for i, (t, ts, res) in enumerate(reversed(st.session_state.history)):
            if st.button(f"{t[:40]}  ·  {ts}", key=f"hist_{i}", use_container_width=True):
                st.session_state.topic = t
                st.session_state.result = res
                st.rerun()
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.caption("No research yet in this session.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("🔎 Multi-Agent AI Research System")
st.caption("Enter a topic and let the agents search, read, write, and critique for you.")

with st.form("research_form", clear_on_submit=False):
    topic_input = st.text_input(
        "Research topic",
        value=st.session_state.topic,
        placeholder="e.g. Latest advances in quantum computing",
    )
    submitted = st.form_submit_button("🚀 Run Research", type="primary")

if submitted:
    topic = topic_input.strip()
    if not topic:
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.topic = topic
        st.session_state.result = None
        with st.status("Starting research…", expanded=True) as status_box:
            try:
                result = run_pipeline(topic, status_box)
                status_box.update(label="Research complete ✅", state="complete", expanded=False)
                st.session_state.result = result
                st.session_state.history.append(
                    (topic, datetime.now().strftime("%H:%M"), result)
                )
            except Exception as e:
                status_box.update(label="Something went wrong ❌", state="error")
                st.error(f"Pipeline failed: {e}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.result
if result:
    st.divider()
    st.subheader(f"Results for: {st.session_state.topic}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🌐 Search Results", "📑 Scraped Content"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "⬇️ Download report (.md)",
            data=result["report"],
            file_name=f"report_{st.session_state.topic[:30].replace(' ', '_')}.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(result["feedback"])

    with tab_search:
        st.markdown(result["search_results"])

    with tab_scraped:
        st.markdown(result["scraped_result"])