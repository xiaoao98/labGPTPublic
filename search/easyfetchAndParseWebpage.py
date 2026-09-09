import requests
from bs4 import BeautifulSoup

def fetch_and_parse_webpage_content(urls):
    """
    Fetches the content from a list of URLs and parses it using BeautifulSoup.

    Args:
        urls (list): A list of URLs (strings) to fetch content from.

    Returns:
        list: A list of dictionaries, where each dictionary contains the 'url'
              and the extracted 'content' (plain text) of the webpage.
              Includes error messages if content could not be retrieved.
    """
    all_page_contents = []
    
    # Common headers to make your request look more like a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "DNT": "1", # Do Not Track request header
        "Connection": "keep-alive"
    }

    for url in urls:
        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status() # Raise an exception for HTTP errors (e.g., 404, 500)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # --- Extracting Content ---
            # There are many ways to get "content". A common approach is to get
            # all visible text. You might need to refine this based on the website's structure.
            # .get_text(separator=' ', strip=True) gets all visible text,
            # separates paragraphs with a space, and strips leading/trailing whitespace.
            # page_text = soup.get_text(separator=' ', strip=True)
            
            # You might want to remove scripts and style elements for cleaner text
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract() # Remove them from the soup
            page_text_clean = soup.get_text(separator=' ', strip=True)

            all_page_contents.append({
                "url": url,
                "content": page_text_clean
            })

        except requests.exceptions.RequestException as e:
            # Handle potential errors during the request (e.g., connection errors, timeouts)
            all_page_contents.append({
                "url": url,
                "content": f"ERROR: Could not retrieve content - {e}"
            })
        except Exception as e:
            # Catch any other unexpected errors during parsing
            all_page_contents.append({
                "url": url,
                "content": f"ERROR: An unexpected error occurred - {e}"
            })

    return all_page_contents

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your actual web links
    my_web_links = [
        "https://www.example.com/page1",
        "https://www.iana.org/domains/example",
        "https://www.python.org/",
        "https://example.invalid-url.xyz/" # An example of a broken/invalid URL
    ]

    contents = fetch_and_parse_webpage_content(my_web_links)

    for i, item in enumerate(contents):
        print(f"\n--- Content from Link {i+1} ---")
        print(f"URL: {item['url']}")
        # Print only the first 500 characters of the content for brevity
        print(f"Content (snippet): {item['content'][:500]}...") 
        print("-" * 30)