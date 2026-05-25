# Multi-Agent Research Assistant

A multi-agent AI system built using the **CrewAI** framework that searches real, validated academic papers from **arXiv** and **OpenAlex**, extracts raw technical data, and automatically generates structured research reports.

This project is tailored specifically to bypass historical background/fluff, focusing strictly on active quantitative matrices and deployment-oriented summaries.

---

## 🛠️ Project Architecture
The framework orchestrates specialized AI agents working sequentially to build production-ready intelligence:
1. **Planner:** Outlines the core technical parameters and tracking metrics for the target domain.
2. **Researcher:** Utilizes search APIs to query arXiv and OpenAlex, strictly filtering for recent academic papers (2024–2026).
3. **Writer:** Compiles and formats raw findings into structural research metrics and deployment-oriented summaries.

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

## 🎬 Live Project Demo
👉 [Watch the Live Demo](https://drive.google.com/file/d/1ZkaS9_WxvelFx0TPR1Fuo-4nQQ-tmF0i/view?usp=drive_link)