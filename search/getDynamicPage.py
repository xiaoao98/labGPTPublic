import time
import requests
from bs4 import BeautifulSoup
from readability import Document 
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def get_dynamic_page_source(url: str) -> str:
    """
    Fetches the full HTML from a URL after JavaScript has rendered, using Selenium.
    This is the "fallback" method for complex pages.
    Includes a 30-second timeout for page loads.
    """
    # print(f"    - Using Selenium for {url}")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    page_source = ""
    try:
        driver.set_page_load_timeout(30)
        
        try:
            driver.get(url)
            time.sleep(5)
            page_source = driver.page_source
        except TimeoutException:
            print(f"    - ❗ ERROR: Selenium timed out after 30 seconds loading {url}")
            return ""
            
    finally:
        driver.quit()
        
    return page_source

def extract_main_content(html: str) -> str:
    """
    Uses the readability library to extract the main article text from HTML.
    """
    doc = Document(html)
    title = doc.title()
    # The summary() method returns the main content as clean HTML
    main_content_html = doc.summary()
    # Use BeautifulSoup to get the plain text from the clean HTML
    soup = BeautifulSoup(main_content_html, 'html.parser')
    main_content_text = soup.get_text(separator=' ', strip=True)
    
    # Combine title and content for a complete result
    return f"{title}\n\n{main_content_text}"


def fetch_all_webpage_content(urls: list[str]) -> list[dict]:
    """
    Takes a list of URLs and returns a list of dictionaries containing the main content.
    """
    all_contents = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for url in urls:
        content = ""
        try:
            # 1. Try the fast method first
            print(f"Processing: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # --- KEY CHANGE HERE: Use readability to extract main content ---
            content = extract_main_content(response.text)
            
            # Heuristic: If content is too short, it might need JS rendering.
            if len(content) < 100:
                # print(f"    - ❌ Content seems minimal. Falling back to Selenium.")
                full_html = get_dynamic_page_source(url)
                if full_html:
                    content = extract_main_content(full_html)
                else:
                    content = "ERROR: Selenium failed to retrieve page source (may have timed out)."
            # else:
                #  print(f"    - ✅ Success with fast method.")
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                # print(f"    - ❗ Received 403 Forbidden. Forcing fallback to Selenium.")
                try:
                    full_html = get_dynamic_page_source(url)
                    if full_html:
                        content = extract_main_content(full_html)
                    else:
                        content = "ERROR: Selenium failed after a 403 error (may have timed out)."
                except Exception as selenium_e:
                    content = f"ERROR: Selenium failed after a 403 error - {selenium_e}"
            else:
                content = f"ERROR: An HTTP error occurred - {e}"
                # print(f"    - ❗ ERROR scraping {url}: {e}")

        except Exception as e:
            content = f"ERROR: A general exception occurred - {e}"
            # print(f"    - ❗ ERROR scraping {url}: {e}")

        all_contents.append({
            "url": url,
            "content": content
        })
        
    return all_contents

if __name__ == '__main__':
    # Before running, you must install the readability library:
    # pip install readability-lxml
    
    urls_to_fetch = [
        "http://books.toscrape.com/",
        "https://example.com/",
        "https://www.theverge.com/2024/2/5/24062753/google-gemini-bard-brand-rename-pro-ultra-1-0-release-date",
        "http://httpbin.org/delay/35", # This URL will time out
    ]

    extracted_contents = fetch_all_webpage_content(urls_to_fetch)

    print("\n" + "="*50)
    print("           FINAL EXTRACTED CONTENT")
    print("="*50)
    for item in extracted_contents:
        print(f"URL: {item['url']}")
        print(f"CONTENT: {item['content'][:500]}...")
        print("-"*50)
