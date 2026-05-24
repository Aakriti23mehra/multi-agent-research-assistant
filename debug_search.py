"""
Run this to debug what's happening:
    python debug_search.py
"""
import requests, urllib.parse, re, time

def test_arxiv(query):
    import xml.etree.ElementTree as ET
    url = (f"https://export.arxiv.org/api/query?"
           f"search_query=all:{urllib.parse.quote(query)}"
           f"&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending")
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return f"arXiv HTTP {r.status_code}"
    root = ET.fromstring(r.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    papers = []
    for e in entries[:3]:
        t = e.find("atom:title", ns)
        p = e.find("atom:published", ns)
        l = e.find("atom:id", ns)
        if t and p:
            papers.append(f"  [{p.text[:4]}] {t.text.strip()[:60]}")
    return f"arXiv: {len(entries)} papers found\n" + "\n".join(papers)

def test_openalex(query):
    r = requests.get(
        "https://api.openalex.org/works",
        params={"search": query, "filter": "publication_year:2020-2026,is_oa:true",
                "per_page": 5, "sort": "relevance_score:desc",
                "select": "title,publication_year"},
        headers={"User-Agent": "ResearchAssistant/1.0 (mailto:test@test.com)"},
        timeout=15
    )
    if r.status_code != 200:
        return f"OpenAlex HTTP {r.status_code}"
    results = r.json().get("results", [])
    papers = [f"  [{p.get('publication_year')}] {p.get('title','')[:60]}" for p in results[:3]]
    return f"OpenAlex: {len(results)} papers found\n" + "\n".join(papers)

def test_core(query):
    r = requests.get(
        "https://api.core.ac.uk/v3/search/works",
        params={"q": query, "limit": 5},
        timeout=15
    )
    if r.status_code != 200:
        return f"CORE HTTP {r.status_code}"
    results = r.json().get("results", [])
    papers = [f"  [{p.get('yearPublished')}] {str(p.get('title',''))[:60]}" for p in results[:3]]
    return f"CORE: {len(results)} papers found\n" + "\n".join(papers)

def test_doaj(query):
    r = requests.get(
        f"https://doaj.org/api/search/articles/{urllib.parse.quote(query)}?pageSize=5",
        timeout=15
    )
    if r.status_code != 200:
        return f"DOAJ HTTP {r.status_code}"
    results = r.json().get("results", [])
    papers = [f"  [{p.get('bibjson',{}).get('year')}] {p.get('bibjson',{}).get('title','')[:60]}" for p in results[:3]]
    return f"DOAJ: {len(results)} papers found\n" + "\n".join(papers)

# Test queries
queries = [
    "RAG retrieval 2024",
    "retrieval augmented generation",
    "RAG LLM",
]

print("="*60)
print("TESTING ALL APIs")
print("="*60)

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-"*40)
    
    try:
        print(test_arxiv(query))
    except Exception as e:
        print(f"arXiv ERROR: {e}")
    
    time.sleep(1)
    
    try:
        print(test_openalex(query))
    except Exception as e:
        print(f"OpenAlex ERROR: {e}")
    
    time.sleep(1)
    
    try:
        print(test_core(query))
    except Exception as e:
        print(f"CORE ERROR: {e}")
    
    time.sleep(1)
    
    try:
        print(test_doaj(query))
    except Exception as e:
        print(f"DOAJ ERROR: {e}")
    
    print()
    time.sleep(2)

print("="*60)
print("Paste this output here so we can fix the issue!")