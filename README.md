# Multi-Agent Research Assistant

A multi-agent AI system built using the **CrewAI** framework that searches real, validated academic papers from **arXiv** and **OpenAlex**, extracts raw technical data, and automatically generates structured research reports.

This project is tailored specifically to bypass historical background and generic summaries, focusing strictly on active quantitative metrics and deployment-oriented research insights.

---

## 🤖 AI Agent & Model Roles

### Planner Agent (Groq LLM)
Outlines the core technical parameters, evaluation criteria, and tracking metrics for the target research domain.

### Researcher Agent (Groq LLM)
Uses search APIs (Serper/CORE) to query **arXiv** and **OpenAlex**, filtering recent academic publications (2024–2026).

### Writer Agent (Groq LLM)
Compiles validated findings into structured research reports and deployment-focused technical summaries.

---

## 🚀 Project Structure

The repository contains configuration files and execution scripts optimized for multi-agent workflows.

- `app.py`  
  Core Streamlit/UI orchestration layer that executes and manages agent workflows.

- `requirements.txt` / `pyproject.toml`  
  Dependency management and environment setup configured for fast execution using **uv**.

- `testing.py`  
  Testing and evaluation script for validating tool outputs and analyzing agent execution sequences.

---

# 🚀 Setup & Installation

## Prerequisites

- Python `>=3.10` and `<3.14`
- **UV** package manager for ultra-fast dependency handling

## 1. Install Dependencies

```bash
pip install uv
crewai install
```

## 2. Run the Application

```bash
python app.py
```

---

## 📄 Demo

[Test video ](https://drive.google.com/file/d/1ZkaS9_WxvelFx0TPR1Fuo-4nQQ-tmF0i/view?usp=sharing)

---