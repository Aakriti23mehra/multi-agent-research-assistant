from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.tools import BaseTool
from dotenv import load_dotenv
import os, re, time
import litellm
import requests
import concurrent.futures

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# =================== Rate Limit Patch =====================================
_original_completion = litellm.completion

def completion_with_retry(*args, **kwargs):
    for attempt in range(4):
        try:
            return _original_completion(*args, **kwargs)
        except litellm.RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"\n⚠️ Rate limit hit. Waiting {wait}s before retry {attempt+1}/4...\n")
            time.sleep(wait)
        except litellm.InternalServerError:
            wait = 15 * (attempt + 1)
            print(f"\n🔌 Server timeout. Waiting {wait}s before retry {attempt+1}/4...\n")
            time.sleep(wait)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}. Retrying {attempt+1}/4...\n")
            time.sleep(10)
    raise RuntimeError("Max retries exceeded.")

litellm.completion = completion_with_retry

# =================== LLM =================================================
llm = LLM(
    model="nvidia_nim/meta/llama-3.1-70b-instruct",
    max_tokens=4096,
    temperature=0.1,
    timeout=90,
)

# =================== SAFE HTTP GET ========================================
def safe_get(url, params=None, headers=None, timeout=20):
    base_headers = {"User-Agent": "ResearchAssistant/1.0 (academic-tool)"}
    if headers:
        base_headers.update(headers)
    for attempt in range(2):
        try:
            r = requests.get(url, params=params,
                             headers=base_headers, timeout=timeout)
            return r
        except requests.exceptions.Timeout:
            if attempt == 0:
                time.sleep(2)
            continue
        except Exception:
            return None
    return None


# =================== RELEVANCE FILTER =====================================
def is_relevant(paper: dict, query: str) -> bool:
    stopwords = {
        "the", "a", "an", "of", "for", "in", "on", "with", "using",
        "based", "and", "or", "to", "is", "are", "system", "systems",
        "2020", "2021", "2022", "2023", "2024", "2025", "2026",
    }
    query_words = [w.lower() for w in re.split(r'\W+', query)
                   if w.lower() not in stopwords and len(w) > 2]

    if not query_words:
        return True

    title    = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()

    # Title match = strong signal → accept immediately
    title_matches = sum(1 for w in query_words if w in title)
    if title_matches >= 1:
        return True

    # Abstract fallback
    matches   = sum(1 for w in query_words if w in abstract)
    threshold = max(1, len(query_words) // 2)
    return matches >= threshold


# =================== ABSTRACT PARSER =====================================
def parse_abstract_fields(abstract: str, title: str) -> dict:
    """
    Extract fields ONLY from what is actually in the abstract.
    NEVER fabricate or infer numbers not present in the text.
    """
    abstract  = (abstract or "").strip()
    sentences = re.split(r'(?<=[.!?])\s+', abstract) if abstract else []

    # Technical Contribution: first 2 sentences of abstract
    contribution = " ".join(sentences[:2]) if sentences else f"See paper: {title}"

    # Key Finding: only sentences that actually contain numbers + metric words
    num_sents = [
        s for s in sentences
        if re.search(
            r'\d+\.?\d*\s*(%|accuracy|F1|BLEU|ROUGE|AUC|AUROC|'
            r'latency|ms\b|NDCG|MRR|Recall|Precision|perplexity)',
            s, re.I)
    ]
    # If no numeric sentence found → use last sentence (conclusion), NOT a number
    key_finding = (num_sents[0] if num_sents
                   else sentences[-1] if len(sentences) > 1
                   else "See paper for findings.")

    # Metrics: ONLY extract numbers actually present in abstract text
    # DO NOT generate or infer any number not explicitly in the abstract
    metrics_raw = re.findall(
        r'(\d+\.?\d*\s*(?:%|accuracy|F1|BLEU|ROUGE|AUC|AUROC|'
        r'latency|ms|NDCG|MRR|Recall|Precision|perplexity)'
        r'[^,.\n]{0,40})',
        abstract, re.I)
    # "See paper" means no metrics in abstract — writer must NOT invent numbers
    metrics = "; ".join(metrics_raw[:3]) if metrics_raw else "See paper"

    # Practical Use: look for code/library mentions in abstract
    lib = re.search(
        r'(github\.com/\S+|pip install \S+|available at \S+)',
        abstract, re.I)
    practical = (
        f"Code available: {lib.group(1)}" if lib
        else f"See paper for implementation details: {title.split(':')[0].strip()}"
    )

    return {
        "technical_contribution": contribution,
        "key_finding":            key_finding,
        "metrics":                metrics,
        "practical_use":          practical,
    }


def format_paper_block(p: dict) -> str:
    f = parse_abstract_fields(p.get("abstract", ""), p.get("title", ""))
    return (
        f"Title: {p['title']}\n"
        f"Authors: {p['authors']}\n"
        f"Year: {p['year']}\n"
        f"Source: {p['url']}\n"
        f"Database: {p['database']}\n"
        f"Technical Contribution: {f['technical_contribution']}\n"
        f"Key Finding: {f['key_finding']}\n"
        f"Metrics: {f['metrics']}\n"
        f"Practical Use: {f['practical_use']}"
    )


# =================== SOURCE 1: arXiv ✅ PRIMARY ===========================
class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Search arXiv for CS/ML/AI papers (2020+). "
        "Best source for RAG, LLM, NLP, CV papers. "
        "Input: 2-4 word query like 'RAG retrieval 2024'"
    )

    def _run(self, query: str) -> list:
        try:
            import urllib.parse, xml.etree.ElementTree as ET
            url = (
                f"https://export.arxiv.org/api/query?"
                f"search_query=all:{urllib.parse.quote(query)}"
                f"&start=0&max_results=15"
                f"&sortBy=submittedDate&sortOrder=descending"
            )
            r = safe_get(url, timeout=20)
            if not r or r.status_code != 200:
                return []

            root = ET.fromstring(r.content)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", ns):
                title_el  = entry.find("atom:title", ns)
                summary   = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)
                link      = entry.find("atom:id", ns)
                authors   = [
                    a.find("atom:name", ns).text
                    for a in entry.findall("atom:author", ns)
                    if a.find("atom:name", ns) is not None
                ]
                if title_el is not None and published is not None:
                    year = published.text[:4]
                    if int(year) >= 2020:
                        p = {
                            "title":    title_el.text.strip(),
                            "authors":  ", ".join(authors[:4]),
                            "year":     year,
                            "url":      (link.text or "N/A").replace("http://", "https://"),
                            "abstract": (summary.text or "").strip()[:600],
                            "database": "arXiv",
                        }
                        if is_relevant(p, query):
                            papers.append(p)
            return papers
        except Exception as e:
            print(f"[arXiv] {e}")
            return []


# =================== SOURCE 2: OpenAlex ✅ PRIMARY ========================
class OpenAlexTool(BaseTool):
    name: str = "openalex_search"
    description: str = (
        "Search OpenAlex (250M papers, free, no key). "
        "Good for recent journal and conference papers. "
        "Input: query string."
    )

    def _run(self, query: str) -> list:
        try:
            r = safe_get(
                "https://api.openalex.org/works",
                params={
                    "search":   query,
                    "filter":   "publication_year:2020-2026,is_oa:true",
                    "per_page": 15,
                    "sort":     "relevance_score:desc",
                    "select":   (
                        "title,authorships,publication_year,doi,"
                        "abstract_inverted_index,primary_location"
                    ),
                },
                headers={
                    "User-Agent": "ResearchAssistant/1.0 (mailto:research@tool.com)"
                },
                timeout=20,
            )
            if not r or r.status_code != 200:
                return []

            papers = []
            for item in r.json().get("results", []):
                year = item.get("publication_year")
                if not year or int(year) < 2020:
                    continue

                title = item.get("title") or "Untitled"
                doi   = item.get("doi") or ""
                url   = (doi if doi.startswith("http")
                         else f"https://doi.org/{doi}" if doi else "")
                if not url:
                    loc = item.get("primary_location") or {}
                    url = (loc.get("landing_page_url")
                           or loc.get("pdf_url") or "N/A")

                inv      = item.get("abstract_inverted_index") or {}
                abstract = ""
                if inv:
                    word_map = {
                        pos: word
                        for word, positions in inv.items()
                        for pos in positions
                    }
                    abstract = " ".join(
                        word_map[i] for i in sorted(word_map)
                    )[:600]

                authors_raw = item.get("authorships") or []
                authors = ", ".join(
                    a.get("author", {}).get("display_name", "")
                    for a in authors_raw[:4]
                )

                p = {
                    "title":    title,
                    "authors":  authors,
                    "year":     str(year),
                    "url":      url,
                    "abstract": abstract,
                    "database": "OpenAlex",
                }
                if is_relevant(p, query):
                    papers.append(p)
            return papers
        except Exception as e:
            print(f"[OpenAlex] {e}")
            return []


# =================== SOURCE 3: DOAJ ⚡ BACKUP =============================
# Tested: sometimes times out (15s), but works with 30s timeout + retry
# Used as backup when arXiv + OpenAlex return fewer than 5 papers
class DoajTool(BaseTool):
    name: str = "doaj_search"
    description: str = (
        "Search DOAJ open-access journals (2020+). "
        "Backup source — used when arXiv and OpenAlex return few papers. "
        "Input: query string."
    )

    def _run(self, query: str) -> list:
        try:
            import urllib.parse
            # Longer timeout (30s) + 2 retries since DOAJ is slow
            r = safe_get(
                f"https://doaj.org/api/search/articles/{urllib.parse.quote(query)}"
                f"?pageSize=10&sort=created_date:desc",
                timeout=30,   # increased from 15 → 30s
            )
            if not r or r.status_code != 200:
                print("[DOAJ] No response or bad status")
                return []

            papers = []
            for item in r.json().get("results", []):
                bib  = item.get("bibjson", {})
                year = bib.get("year")
                if not year or int(year) < 2020:
                    continue

                doi   = next(
                    (i.get("id") for i in bib.get("identifier", [])
                     if i.get("type") == "doi"),
                    None
                )
                links = bib.get("link", [])
                url   = (f"https://doi.org/{doi}" if doi
                         else links[0].get("url") if links else "N/A")

                p = {
                    "title":    bib.get("title", "Untitled"),
                    "authors":  ", ".join(
                        a.get("name", "")
                        for a in bib.get("author", [])[:4]
                    ),
                    "year":     str(year),
                    "url":      url,
                    "abstract": (bib.get("abstract") or "")[:600],
                    "database": "DOAJ",
                }
                if is_relevant(p, query):
                    papers.append(p)
            return papers
        except Exception as e:
            print(f"[DOAJ] {e}")
            return []


# =================== SOURCE 4: Semantic Scholar (optional key) ============
class SemanticScholarTool(BaseTool):
    name: str = "semantic_scholar_search"
    description: str = (
        "Search Semantic Scholar. Needs S2_API_KEY in .env. "
        "Auto-skipped if no key present. Input: query string."
    )

    def _run(self, query: str) -> list:
        try:
            s2_key = os.getenv("S2_API_KEY", "")
            if not s2_key:
                print("[S2] No S2_API_KEY — skipping.")
                return []
            r = safe_get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query":  query,
                    "limit":  10,
                    "fields": "title,authors,year,abstract,openAccessPdf,paperId",
                },
                headers={"x-api-key": s2_key},
                timeout=20,
            )
            if not r or r.status_code != 200:
                return []
            papers = []
            for p in r.json().get("data", []):
                year = p.get("year")
                if year and int(year) >= 2020:
                    pdf = p.get("openAccessPdf")
                    url = ((pdf.get("url") if pdf else None)
                           or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}")
                    paper = {
                        "title":    p.get("title", "Untitled"),
                        "authors":  ", ".join(
                            a.get("name", "")
                            for a in p.get("authors", [])[:4]
                        ),
                        "year":     str(year),
                        "url":      url,
                        "abstract": (p.get("abstract") or "")[:600],
                        "database": "Semantic Scholar",
                    }
                    if is_relevant(paper, query):
                        papers.append(paper)
            return papers
        except Exception as e:
            print(f"[S2] {e}")
            return []


# =================== MULTI-SOURCE ORCHESTRATOR ===========================
# Strategy:
#   PRIMARY:  arXiv + OpenAlex (fast, reliable, relevant)
#   BACKUP:   DOAJ (slower, used only if primary returns < 5 papers)
#   OPTIONAL: Semantic Scholar (needs S2_API_KEY)
# REMOVED: CORE (irrelevant old papers), CrossRef (wrong domain results)
class MultiSourceSearchTool(BaseTool):
    name: str = "multi_source_search"
    description: str = (
        "PRIMARY TOOL — CALL THIS FIRST. "
        "Searches arXiv and OpenAlex (primary) + DOAJ as backup. "
        "Only returns papers relevant to the query. "
        "Returns pre-formatted blocks with ALL 9 fields filled. "
        "COPY the output directly — do not rewrite. "
        "Input: short 2-4 word query like 'RAG retrieval 2024'"
    )

    def _run(self, query: str) -> str:

        # ── PHASE 1: Primary sources (fast) ───────────────────────
        primary_sources = {
            "arXiv":            ArxivSearchTool(),
            "OpenAlex":         OpenAlexTool(),
            "Semantic Scholar": SemanticScholarTool(),  # skipped if no key
        }

        all_papers    = []
        source_counts = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(t._run, query): n
                for n, t in primary_sources.items()
            }
            for future in concurrent.futures.as_completed(futures):
                src = futures[future]
                try:
                    result = future.result(timeout=25)
                    if isinstance(result, list):
                        all_papers.extend(result)
                        source_counts[src] = len(result)
                    else:
                        source_counts[src] = 0
                except Exception:
                    source_counts[src] = 0

        # ── PHASE 2: DOAJ backup (only if primary found < 5 papers) ──
        primary_count = len(all_papers)
        if primary_count < 5:
            print(f"[MultiSource] Primary found {primary_count} papers — trying DOAJ backup...")
            try:
                doaj_papers = DoajTool()._run(query)
                if isinstance(doaj_papers, list) and doaj_papers:
                    all_papers.extend(doaj_papers)
                    source_counts["DOAJ"] = len(doaj_papers)
                    print(f"[MultiSource] DOAJ added {len(doaj_papers)} papers")
                else:
                    source_counts["DOAJ"] = 0
            except Exception as e:
                print(f"[MultiSource] DOAJ backup failed: {e}")
                source_counts["DOAJ"] = 0
        else:
            source_counts["DOAJ"] = 0  # not needed

        # ── Deduplicate by title ───────────────────────────────────
        seen, unique = set(), []
        for p in all_papers:
            key = p["title"].lower().strip()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(p)

        # ── Sort: newest first, prefer arXiv/S2 ───────────────────
        db_priority = {
            "arXiv": 4, "Semantic Scholar": 4,
            "OpenAlex": 3, "DOAJ": 1,
        }
        unique.sort(
            key=lambda p: (
                int(p.get("year", 0)),
                db_priority.get(p["database"], 0),
            ),
            reverse=True,
        )

        counts_str = " | ".join(
            f"{s}={c}" for s, c in source_counts.items() if c > 0
        )

        if not unique:
            all_counts = " | ".join(
                f"{s}={c}" for s, c in source_counts.items()
            )
            return (
                f"NO_PAPERS_FOUND for query='{query}'. "
                f"Sources: {all_counts}. "
                f"Try a 2-word query without year."
            )

        blocks = [format_paper_block(p) for p in unique[:10]]
        header = (
            f"FOUND {len(unique)} relevant papers | {counts_str}\n"
            f"COPY THESE BLOCKS DIRECTLY — do not rewrite.\n\n"
        )
        return header + "\n\n---\n\n".join(blocks)


# =================== Tool Instances =======================================
multi_source_search   = MultiSourceSearchTool()
arxiv_tool            = ArxivSearchTool()
openalex_tool         = OpenAlexTool()
doaj_tool             = DoajTool()
semantic_scholar_tool = SemanticScholarTool()

web_search = SerperDevTool(
    name="search",
    description=(
        "Last-resort web search. ONLY if multi_source_search returns NO_PAPERS_FOUND. "
        "Query: topic + site:arxiv.org + year. Example: 'RAG retrieval site:arxiv.org 2024'"
    )
)
web_scrape = ScrapeWebsiteTool(name="scrape")


# =================== GUARDRAIL ===========================================
# MINIMAL guardrail — just check if ANY paper content exists
# Strict guardrail was causing infinite loops
def validate_research(result):
    raw = result.raw

    # Only fail if output is completely empty or just an error message
    if not raw or len(raw.strip()) < 50:
        return (False, (
            "Output is empty. Call multi_source_search('RAG retrieval') "
            "and paste the results directly."
        ))

    # If tool found nothing and agent gave up
    if "NO_PAPERS_FOUND" in raw and "Title:" not in raw and "http" not in raw:
        return (False, (
            "No papers found. Try: arxiv_search('retrieval augmented generation') "
            "or openalex_search('RAG LLM')"
        ))

    # Pass everything else — writer will work with whatever researcher found
    return (True, raw)


# =================== Crew ================================================
@CrewBase
class ResearchAssistantCrew():
    agents: List[BaseAgent]
    tasks:  List[Task]
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def planner_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["planner_agent"],
            llm=llm, verbose=True, max_iter=2,
            respect_context_window=True,
        )

    @agent
    def researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher_agent"],
            llm=llm,
            tools=[
                multi_source_search,    # PRIMARY — arXiv+OpenAlex, DOAJ backup
                arxiv_tool,             # individual fallbacks
                openalex_tool,
                doaj_tool,              # individual DOAJ fallback
                semantic_scholar_tool,  # needs S2_API_KEY
                web_search,             # last resort
                web_scrape,
            ],
            verbose=True,
            max_iter=7,
            respect_context_window=True,
        )

    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["writer_agent"],
            llm=llm, verbose=True, max_iter=2,
            respect_context_window=True,
        )

    @agent
    def review_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["review_agent"],
            llm=llm, verbose=True,
            max_iter=1,
            respect_context_window=True,
        )

    @task
    def planning_task(self) -> Task:
        return Task(config=self.tasks_config["planning_task"])

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            context=[self.planning_task()],
            guardrail=validate_research,
            guardrail_max_retries=1,
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["writing_task"],
            context=[self.research_task()],
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_task"],
            context=[self.writing_task()],
            output_file="final_report.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            max_rpm=15,
        )