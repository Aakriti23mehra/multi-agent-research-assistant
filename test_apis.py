"""
Run this script on YOUR machine to check which APIs are accessible:
    python test_apis.py
"""
import requests, urllib.parse, time

def test(name, url, params=None, headers=None):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 200:
            print(f"✅ {name}: WORKS (HTTP 200)")
            return True
        else:
            print(f"❌ {name}: HTTP {r.status_code} — {r.text[:80]}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ {name}: TIMEOUT (>15s)")
        return False
    except Exception as e:
        print(f"❌ {name}: ERROR — {e}")
        return False

print("="*60)
print("Testing all academic APIs from YOUR machine...")
print("="*60)

test("arXiv",
     "https://export.arxiv.org/api/query",
     params={"search_query": "all:RAG", "max_results": 2})

time.sleep(1)

test("Semantic Scholar",
     "https://api.semanticscholar.org/graph/v1/paper/search",
     params={"query": "RAG", "limit": 2, "fields": "title,year"},
     headers={"User-Agent": "ResearchTool/1.0"})

time.sleep(1)

test("OpenAlex",
     "https://api.openalex.org/works",
     params={"search": "RAG", "per_page": 2},
     headers={"User-Agent": "ResearchTool/1.0 (mailto:test@test.com)"})

time.sleep(1)

test("CrossRef",
     "https://api.crossref.org/works",
     params={"query": "retrieval augmented generation", "rows": 2},
     headers={"User-Agent": "ResearchTool/1.0"})

time.sleep(1)

test("CORE (no key)",
     "https://api.core.ac.uk/v3/search/works",
     params={"q": "RAG", "limit": 2})

time.sleep(1)

test("DOAJ",
     "https://doaj.org/api/search/articles/RAG",
     params={"pageSize": 2})

print("="*60)
print("Share these results so we can fix the blocked ones!")