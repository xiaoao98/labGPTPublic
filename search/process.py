from search.easyfetchAndParseWebpage import fetch_and_parse_webpage_content
from getDynamicPage import fetch_all_webpage_content
from duckducksearch import get_useful_link

query = "What is the Vance Lab"
links, snippets = get_useful_link(query)
all_page_contents = fetch_all_webpage_content(links)
for i, item in enumerate(all_page_contents):
        print(f"\n--- Content from Link {i+1} ---")
        print(f"URL: {item['url']}")
        print(f"Snippet: {snippets[i]}")
        print(f"Content: {item['content'][:5000]}") 
        print("-" * 30)