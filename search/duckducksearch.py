from ddgs import DDGS

ddgs = DDGS()

def search_internet(query):
    try:
        results = ddgs.text(query, max_results=5)
        # Filter out irrelevant results
        filtered_results = [result for result in results if 'body' in result]
        return filtered_results
    except Exception as e:
        print(f"Error searching internet: {e}")
        return []

def get_useful_link(query):
    # Preprocess user input
    query = query.strip().lower()
    # print("Question: " + query)
    # Perform internet search
    search_results = search_internet(query)
    # print("Search Results:")
    links = []
    snippets = []
    for result in search_results:
    #     print(f"Title: {result['title']}")
    #     print(f"URL: {result['href']}")
    #     print(f"Snippet: {result['body']}\n")
        snippets.append(result['body'])
        links.append(result['href'])
    return links, snippets

# query = "What is the recommended tour place in Houston in summer"
# links = get_useful_link(query)

# # Print the search results and the AI response


# print("--------------------------------------------------------")
# print(links)