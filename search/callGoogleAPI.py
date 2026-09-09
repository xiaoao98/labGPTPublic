"""Google Custom Search wrapper.

Credentials come from the environment, never from source. Set GOOGLE_API_KEY and
GOOGLE_SEARCH_ENGINE_ID in a .env file (see .env.example), which .gitignore excludes.

Not currently wired into demo.py, which uses search/duckducksearch.py. Kept as a working
alternative because ddgs gets rate limited under load.
"""

import os

import requests

SEARCH_URL = "https://www.googleapis.com/customsearch/v1"


class MissingCredentials(RuntimeError):
    pass


def _credentials():
    api_key = os.environ.get("GOOGLE_API_KEY")
    engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
    missing = [
        name
        for name, value in (
            ("GOOGLE_API_KEY", api_key),
            ("GOOGLE_SEARCH_ENGINE_ID", engine_id),
        )
        if not value
    ]
    if missing:
        raise MissingCredentials(
            f"missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill it in."
        )
    return api_key, engine_id


def search(query, max_results=3):
    """Return a list of {title, link, snippet} dicts for a query."""
    api_key, engine_id = _credentials()
    params = {"key": api_key, "cx": engine_id, "q": query}

    try:
        response = requests.get(SEARCH_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"API request failed: {exc}")
        return []

    items = response.json().get("items", [])
    return [
        {
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
        }
        for item in items[:max_results]
    ]


def get_useful_link(query):
    """Same shape as search/duckducksearch.get_useful_link, so the two are swappable."""
    results = search(query)
    links = [r["link"] for r in results if r["link"]]
    snippets = [r["snippet"] or "" for r in results if r["link"]]
    return links, snippets


if __name__ == "__main__":
    for index, result in enumerate(search("How is the weather in Houston today"), start=1):
        print(f"Result {index}:")
        print(f"  Title: {result['title']}")
        print(f"  Link: {result['link']}")
        print(f"  Snippet: {result['snippet']}")
        print("-" * 20)
