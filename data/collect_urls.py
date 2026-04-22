import re, time, random
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
TOPICS = [
    "politics", "election", "congress", "senate", "white house", "president",
    "economy", "inflation", "jobs", "federal reserve", "tariff", "trade",
    "health", "cancer", "mental health", "vaccine", "drug",
    "crime", "shooting", "murder", "police", "court", "trial", "prison",
    "climate", "weather", "hurricane", "wildfire", "earthquake", "flood",
    "technology", "ai", "artificial intelligence", "apple", "google", "microsoft",
    "ukraine", "russia", "israel", "iran", "china", "north korea", "nato",
    "immigration", "border", "deportation", "asylum", "migrant",
    "education", "school", "student", "university", "college",
    "sports", "nfl", "nba", "olympics", "world cup", "baseball",
    "entertainment", "celebrity", "movie", "music", "television",
    "business", "stock market", "recession", "bank", "housing",
    "race", "civil rights", "lgbtq", "abortion", "gun control",
    "california", "texas", "new york", "florida", "supreme court",
    "disaster", "accident", "explosion", "fire", "crash",
    "investigation", "fbi", "cia", "national security",
    "trump", "biden", "harris", "democrat", "republican",
    "protest", "riot", "lawsuit", "indictment", "verdict",
    # sports (expanded)
    "soccer", "tennis", "golf", "hockey", "ncaa", "super bowl", "draft", "trade deadline",
    "nhl", "mls", "ufc", "boxing", "marathon", "swimming", "gymnastics",
    # entertainment (expanded)
    "netflix", "streaming", "oscar", "grammy", "emmy", "box office", "award",
    "disney", "hollywood", "podcast", "book", "broadway",
    # technology (expanded)
    "cybersecurity", "hack", "data breach", "tesla", "amazon", "meta", "facebook",
    "twitter", "social media", "cryptocurrency", "bitcoin", "spacex", "nasa", "space",
    "electric vehicle", "self-driving", "robotics", "chip", "semiconductor",
    # health (expanded)
    "covid", "pandemic", "obesity", "alzheimer", "opioid", "fentanyl", "hospital",
    "insurance", "medicare", "medicaid", "aca", "diabetes", "heart disease", "flu",
    "overdose", "addiction", "abortion pill", "ivf", "therapy",
    # environment
    "environment", "pollution", "carbon", "nuclear", "solar", "wind energy",
    "plastic", "endangered species", "deforestation", "oil spill", "emissions",
    # economy (expanded)
    "mortgage", "debt", "deficit", "oil", "gas prices", "unemployment", "layoff",
    "minimum wage", "poverty", "homelessness", "rent", "cryptocurrency crash",
    # international (expanded)
    "middle east", "taiwan", "india", "pakistan", "mexico", "europe", "uk",
    "africa", "australia", "japan", "south korea", "venezuela", "cuba",
    "saudi arabia", "afghanistan", "war", "ceasefire", "sanctions",
    # social issues (expanded)
    "transgender", "affirmative action", "hate crime", "police brutality",
    "voting rights", "gerrymandering", "free speech", "censorship",
    # government / agencies
    "pentagon", "justice department", "treasury", "homeland security",
    "executive order", "veto", "impeachment", "cabinet", "budget",
    "military", "veteran", "draft", "drone", "missile", "nuclear weapon",
    # legal
    "constitution", "first amendment", "second amendment", "pardon", "appeal",
    "class action", "settlement", "subpoena", "contempt",
    # additional states / cities
    "chicago", "los angeles", "seattle", "miami", "atlanta", "houston",
    "las vegas", "new orleans", "phoenix", "denver",
]

def scrape(site, source_keyword, source_suffix_pattern, label, out_csv):
    suffix_re = re.compile(source_suffix_pattern, re.IGNORECASE)
    rows = []
    seen = set()
    for i, topic in enumerate(TOPICS):
        query = quote(f"{topic} site:{site}")
        feed_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        time.sleep(random.uniform(0.8, 1.8))
        try:
            req = Request(feed_url, headers=HEADERS)
            xml = urlopen(req, timeout=20).read().decode("utf-8", errors="replace")
            root = ET.fromstring(re.sub(r'\s+xmlns[^"]*"[^"]*"', "", xml))
            for item in root.iter("item"):
                title_el  = item.find("title")
                source_el = item.find("source")
                if title_el is None or not title_el.text:
                    continue
                if source_el is None or source_keyword not in (source_el.text or "").lower():
                    continue
                t = suffix_re.sub("", title_el.text).strip().lower()
                if not t or t in seen:
                    continue
                seen.add(t)
                rows.append({"url": "", "headline": t, "label": label})
            print(f"[{i+1}/{len(TOPICS)}] {topic} — {len(rows)} total so far")
        except Exception as e:
            print(f"[{i+1}/{len(TOPICS)}] FAIL {topic}: {e}")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv}")

print("=== NBC News ===")
scrape(
    site="nbcnews.com",
    source_keyword="nbc",
    source_suffix_pattern=r"\s*[|\-–—]\s*(nbc news|msnbc).*$",
    label=1,
    out_csv="gnews_headlines_nbc.csv",
)

print("=== Fox News ===")
scrape(
    site="foxnews.com",
    source_keyword="fox",
    source_suffix_pattern=r"\s*[|\-–—]\s*fox news.*$",
    label=0,
    out_csv="gnews_headlines_fox.csv",
)
