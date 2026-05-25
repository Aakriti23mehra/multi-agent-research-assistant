# Multi-Agent Research Assistant

A multi-agent AI system built using the **CrewAI** framework that searches real, validated academic papers from **arXiv** and **OpenAlex**, extracts raw technical data, and automatically generates structured research reports.

This project is tailored specifically to bypass historical background/fluff, focusing strictly on active quantitative matrices and deployment-oriented summaries.

---

### 🤖 AI Agent & Model Roles:
* **Planner Agent (Groq LLM):** Outlines the core technical parameters and tracking metrics for the target domain.
* **Researcher Agent (Groq LLM):** Utilizes search APIs (Serper/CORE) to query arXiv and OpenAlex, strictly filtering for recent academic papers (2024–2026).
* **Writer Agent (Groq LLM):** Compiles and formats raw findings into structural research metrics and deployment-oriented summaries.

---

### 🚀 Project Structure

The repository contains configuration files and core execution scripts tailored for multi-agent workflows:

* `app.py` - Core Streamlit/UI or orchestration script designed to execute agent workflows and handle local application layers.
* `requirements.txt` / `pyproject.toml` - Dependency configuration mapped for fast execution using the `uv` package manager.
* `testing.py` - Evaluation script designed to run test inferences, validate tool outputs, and analyze agent sequence execution.

---

## 🚀 Setup & Installation

### Prerequisites
* Python `>=3.10` and `<3.14`
* **UV** package manager (for ultra-fast dependency management)

### 1. Install Dependencies
Sabse pehle `uv` install karein aur project package dependencies ko lock karein:

```bash
pip install uv
crewai install