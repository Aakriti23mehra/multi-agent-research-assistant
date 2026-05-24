#!/usr/bin/env python
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# ── Import from correct path ──────────────────────────────────────
from src.multi_agent_research_assistant.crew import ResearchAssistantCrew

def run():
    """Main entry point for the Multi-Agent Research Assistant Pipeline"""

    inputs = {
        'topic': 'High-performance Image Caption Generation Architectures',
        'current_year': str(datetime.now().year)
    }

    print("\n" + "="*60)
    print("⚡  MULTI-AGENT RESEARCH ASSISTANT")
    print("⚡  NVIDIA NIM + Llama 3.1 70B + arXiv API")
    print("="*60)
    print(f"📌 Topic   : {inputs['topic']}")
    print(f"📅 Year    : {inputs['current_year']}")
    print(f"🤖 Agents  : Planner → Researcher → Writer → Reviewer")
    print("="*60 + "\n")

    try:
        result = ResearchAssistantCrew().crew().kickoff(inputs=inputs)

        print("\n" + "="*60)
        print("✅  PIPELINE COMPLETE — REPORT GENERATED")
        print("="*60 + "\n")
        print(result)

        # Also save to file
        output_path = "final_report.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(result))
        print(f"\n📄 Report saved to: {output_path}")

    except Exception as e:
        print("\n" + "="*60)
        print("❌  PIPELINE EXECUTION FAILED")
        print("="*60)
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    run()