import sys
import os
import time
import queue
import threading

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Research Crew")
    st.markdown("""
    **Multi-Agent System**
    - 🎯 **Planner:** Technical Roadmap
    - 🔍 **Researcher:** arXiv · Semantic Scholar · CORE · DOAJ
    - ✍️ **Writer:** Report Generation
    - 🧐 **Reviewer:** Fact Checking
    """)
    st.divider()
    st.markdown("**Academic Sources**")
    st.markdown("""
    | Source | Coverage |
    |---|---|
    | arXiv | Preprints |
    | Semantic Scholar | CS + Science |
    | CORE | Open Access |
    | DOAJ | Journals |
    """)
    st.divider()
    st.info("Powered by NVIDIA NIM + Llama 3.1 70B")

# ── Main UI ────────────────────────────────────────────────────────────────
st.title("🔬 Multi-Agent Academic Research Assistant")
st.subheader("Generate structured technical reports from recent academic research (2020–2026).")
st.caption("Sources searched simultaneously: arXiv · Semantic Scholar · CORE · DOAJ")

topic = st.text_input(
    "Enter your research topic:",
    placeholder="e.g., Retrieval-Augmented Generation Systems..."
)

col1, col2 = st.columns([1, 5])

if col1.button("🚀 Start Research", type="primary"):
    if not topic:
        st.warning("Please enter a topic to begin.")
    else:
        st.divider()

        log_queue = queue.Queue()

        st.markdown("### 📊 Agent Pipeline Status")
        status_area = st.empty()

        st.markdown("### 🔴 Live Activity Log")
        log_area = st.empty()

        agent_states = {
            "🎯 Planner":    "⏳ Waiting",
            "🔍 Researcher": "⏳ Waiting",
            "✍️ Writer":     "⏳ Waiting",
            "🧐 Reviewer":   "⏳ Waiting",
        }

        def render_status(states):
            rows = "| Agent | Status |\n|---|---|\n"
            for agent, state in states.items():
                rows += f"| {agent} | {state} |\n"
            status_area.markdown(rows)

        render_status(agent_states)

        # ── Crew runner (background thread) ───────────────────────
        def run_crew():
            import io
            from contextlib import redirect_stdout
            from src.multi_agent_research_assistant.crew import ResearchAssistantCrew

            class QueueLogger(io.StringIO):
                def write(self, text):
                    if text.strip():
                        log_queue.put(("log", text))

            try:
                inputs = {
                    'topic': topic,
                    'current_year': str(datetime.now().year)
                }
                logger = QueueLogger()
                with redirect_stdout(logger):
                    crew_instance = ResearchAssistantCrew().crew()
                    result = crew_instance.kickoff(inputs=inputs)
                log_queue.put(("done", result.raw))
            except Exception as e:
                log_queue.put(("error", str(e)))

        thread = threading.Thread(target=run_crew, daemon=True)
        thread.start()

        log_messages  = []
        report_content = None
        error_msg      = None
        running        = True

        # ── One-time flags ─────────────────────────────────────────
        flags = {
            "planner_started":         False,
            "planner_done":            False,
            "researcher_started":      False,
            "multi_source_called":     False,   # fires when any academic tool is called
            "arxiv_individual":        False,   # fires if arxiv_search called individually
            "semantic_scholar_called": False,
            "core_called":             False,
            "doaj_called":             False,
            "web_searched":            False,
            "scraped":                 False,
            "guardrail_failed":        False,
            "guardrail_passed":        False,
            "researcher_done":         False,
            "writer_started":          False,
            "writer_done":             False,
            "reviewer_started":        False,
        }

        def add_log(msg):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_messages.append(f"`{timestamp}` {msg}")
            log_area.markdown("\n\n".join(log_messages[-20:]))

        add_log(f"🚀 Starting research on: **{topic}**")
        agent_states["🎯 Planner"] = "🔄 Running..."
        render_status(agent_states)

        # ── Main polling loop ──────────────────────────────────────
        while running:
            try:
                while True:
                    msg_type, msg_data = log_queue.get_nowait()

                    # ── Terminal messages ──────────────────────────
                    if msg_type == "done":
                        report_content = msg_data
                        for ag in agent_states:
                            agent_states[ag] = "✅ Done"
                        render_status(agent_states)
                        add_log("🎉 **All agents completed! Generating report...**")
                        running = False
                        break

                    elif msg_type == "error":
                        error_msg = msg_data
                        for ag in agent_states:
                            if agent_states[ag] == "🔄 Running...":
                                agent_states[ag] = "❌ Error"
                        render_status(agent_states)
                        add_log(f"❌ **Error:** {msg_data[:200]}")
                        running = False
                        break

                    # ── Log-line parsing ───────────────────────────
                    elif msg_type == "log":
                        text = msg_data

                        # Planner started
                        if not flags["planner_started"] and (
                            "Research Strategist" in text or
                            "planning_task" in text.lower()
                        ):
                            flags["planner_started"] = True
                            agent_states["🎯 Planner"] = "🔄 Running..."
                            render_status(agent_states)
                            add_log("🎯 **Planner** is creating technical research roadmap...")

                        # Planner done
                        if not flags["planner_done"] and (
                            "planning_task" in text.lower() and
                            "completed" in text.lower()
                        ):
                            flags["planner_done"] = True
                            agent_states["🎯 Planner"] = "✅ Done"
                            render_status(agent_states)
                            add_log("🎯 **Planner** finished roadmap!")

                        # Researcher started
                        if not flags["researcher_started"] and (
                            "Academic Researcher" in text or
                            "research_task" in text.lower()
                        ):
                            flags["researcher_started"] = True
                            agent_states["🎯 Planner"]    = "✅ Done"
                            agent_states["🔍 Researcher"] = "🔄 Running..."
                            render_status(agent_states)
                            add_log(
                                "🔍 **Researcher** searching "
                                "arXiv · Semantic Scholar · CORE · DOAJ..."
                            )

                        # Multi-source search called (PRIMARY tool)
                        if not flags["multi_source_called"] and (
                            "multi_source_search" in text.lower()
                        ):
                            flags["multi_source_called"] = True
                            add_log(
                                "📡 Querying **4 sources in parallel** "
                                "(arXiv · Semantic Scholar · CORE · DOAJ)..."
                            )

                        # Individual fallback — arXiv
                        if not flags["arxiv_individual"] and (
                            "arxiv_search" in text.lower() and
                            "multi_source" not in text.lower()
                        ):
                            flags["arxiv_individual"] = True
                            add_log("📄 Fallback: querying **arXiv** individually...")

                        # Individual fallback — Semantic Scholar
                        if not flags["semantic_scholar_called"] and (
                            "semantic_scholar_search" in text.lower() and
                            "multi_source" not in text.lower()
                        ):
                            flags["semantic_scholar_called"] = True
                            add_log("📄 Fallback: querying **Semantic Scholar** individually...")

                        # Individual fallback — CORE
                        if not flags["core_called"] and (
                            "core_search" in text.lower() and
                            "multi_source" not in text.lower()
                        ):
                            flags["core_called"] = True
                            add_log("📄 Fallback: querying **CORE** individually...")

                        # Individual fallback — DOAJ
                        if not flags["doaj_called"] and (
                            "doaj_search" in text.lower() and
                            "multi_source" not in text.lower()
                        ):
                            flags["doaj_called"] = True
                            add_log("📄 Fallback: querying **DOAJ** individually...")

                        # Web search (last resort)
                        if not flags["web_searched"] and (
                            "Tool: search" in text and "Started" in text
                        ):
                            flags["web_searched"] = True
                            add_log("🔎 Last resort: running **web search** for papers...")

                        # Scrape
                        if not flags["scraped"] and (
                            "Tool: scrape" in text and "Started" in text
                        ):
                            flags["scraped"] = True
                            add_log("📄 Reading full paper content from academic URL...")

                        # Guardrail failed
                        if not flags["guardrail_failed"] and (
                            "Guardrail" in text and (
                                "Failed" in text or
                                "INSUFFICIENT" in text or
                                "retries" in text.lower()
                            )
                        ):
                            flags["guardrail_failed"] = True
                            add_log(
                                "⚠️ Validation failed — retrying with a broader query..."
                            )

                        # Guardrail passed
                        if not flags["guardrail_passed"] and (
                            "Guardrail Passed" in text or
                            "Validation Successful" in text
                        ):
                            flags["guardrail_passed"] = True
                            add_log("🛡️ Research validated — 3+ real papers confirmed!")

                        # Researcher done
                        if not flags["researcher_done"] and (
                            "research_task" in text.lower() and
                            "completed" in text.lower()
                        ):
                            flags["researcher_done"] = True
                            agent_states["🔍 Researcher"] = "✅ Done"
                            render_status(agent_states)
                            add_log("🔍 **Researcher** complete — papers ready for Writer!")

                        # Writer started
                        if not flags["writer_started"] and (
                            "Academic Report Writer" in text or
                            "writing_task" in text.lower()
                        ):
                            flags["writer_started"] = True
                            agent_states["🔍 Researcher"] = "✅ Done"
                            agent_states["✍️ Writer"]     = "🔄 Running..."
                            render_status(agent_states)
                            add_log(
                                "✍️ **Writer** is generating technical report "
                                "with benchmarking table..."
                            )

                        # Writer done
                        if not flags["writer_done"] and (
                            "writing_task" in text.lower() and
                            "completed" in text.lower()
                        ):
                            flags["writer_done"] = True
                            agent_states["✍️ Writer"] = "✅ Done"
                            render_status(agent_states)
                            add_log("✍️ **Writer** finished the report draft!")

                        # Reviewer started
                        if not flags["reviewer_started"] and (
                            "Academic Editor" in text or
                            "review_task" in text.lower()
                        ):
                            flags["reviewer_started"] = True
                            agent_states["✍️ Writer"]   = "✅ Done"
                            agent_states["🧐 Reviewer"] = "🔄 Running..."
                            render_status(agent_states)
                            add_log(
                                "🧐 **Reviewer** is verifying citations and formatting..."
                            )

                        # Rate limit / timeout
                        if "Rate limit" in text:
                            add_log(
                                "⚠️ Rate limit hit — waiting and retrying automatically..."
                            )

                        if "Server timeout" in text:
                            add_log("🔌 Server timeout — retrying automatically...")

            except queue.Empty:
                pass

            if running:
                time.sleep(0.3)

        # ── Final Report ───────────────────────────────────────────
        if report_content:
            st.divider()
            st.success("✅ Research Complete!")
            st.markdown("### 📄 Final Technical Research Report")

            report_placeholder = st.empty()
            streamed_text = ""

            for word in report_content.split(" "):
                streamed_text += word + " "
                report_placeholder.markdown(streamed_text + "▌")
                time.sleep(0.02)

            report_placeholder.markdown(streamed_text)

            st.divider()
            st.download_button(
                label="⬇️ Download Report (.md)",
                data=report_content,
                file_name=f"Research_Report_{topic.replace(' ', '_')}.md",
                mime="text/markdown",
            )

        elif error_msg:
            st.error(f"An error occurred: {error_msg}")

else:
    if not topic:
        st.info("Enter a topic and click '🚀 Start Research' to begin.")

st.divider()
st.caption(
    f"© {datetime.now().year} Multi-Agent Research Assistant | "
    "Powered by NVIDIA NIM + Llama 3.1 70B | "
    "Sources: arXiv · Semantic Scholar · CORE · DOAJ"
)