import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup
import threading
import asyncio

@dataclass
class Task:
    """Represents a task to be processed by the learning agent"""
    id: str
    description: str
    status: str = "new"
    context: Dict = None
    steps: List[str] = None
    result: Optional[str] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.steps is None:
            self.steps = []

class SearchResult:
    """Represents a single search result with proper URL handling"""
    def __init__(self, title: str, url: str, snippet: str):
        self.title = self._clean_text(title)
        self.url = self._clean_url(url)
        self.snippet = self._clean_text(snippet)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('"', '"').replace('"', '"')
        return text

    def _clean_url(self, url: str) -> str:
        """Clean and validate URLs"""
        if not url:
            return ""
        url = url.split('?')[0]
        url = re.sub(r'/amp$', '', url)
        return url.strip()

    def is_valid(self) -> bool:
        """Check if the search result is valid"""
        return bool(
            self.title and 
            self.url and 
            not self.url.startswith('https://duckduckgo.com') and
            not self.url.startswith('https://www.facebook.com') and
            not self.url.startswith('https://twitter.com')
        )

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet
        }

class DuckDuckGoSearch:
    """Handles web searches using DuckDuckGo with improved reliability"""
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.lock = threading.Lock()
        self._init_connection_pool()

    def _init_connection_pool(self):
        """Initialize connection pool settings"""
        import urllib3
        urllib3.PoolManager(maxsize=10, retries=urllib3.Retry(3))
        # Set Firefox capabilities for better connection handling
        self.firefox_capabilities = webdriver.FirefoxOptions()
        self.firefox_capabilities.add_argument('--headless')
        self.firefox_capabilities.add_argument('--disable-gpu')
        self.firefox_capabilities.add_argument('--no-sandbox')
        self.firefox_capabilities.add_argument('--disable-dev-shm-usage')
        self.firefox_capabilities.add_argument('--disable-notifications')
        self.firefox_capabilities.add_argument('--disable-popup-blocking')
        self.firefox_capabilities.set_preference("browser.download.folderList", 2)
        self.firefox_capabilities.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
        # Add connection pooling settings
        self.firefox_capabilities.set_preference("network.http.connection-timeout", 10)
        self.firefox_capabilities.set_preference("network.http.max-connections", 10)
        self.firefox_capabilities.set_preference("network.http.max-persistent-connections-per-server", 5)

    def initialize_browser(self):
        """Initialize browser with improved error handling"""
        with self.lock:
            try:
                driver = webdriver.Firefox(options=self.firefox_capabilities)
                driver.set_page_load_timeout(20)
                driver.implicitly_wait(5)
                return driver
            except WebDriverException as e:
                logger.error(f"Failed to initialize Firefox: {e}")
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def search(self, query: str) -> List[SearchResult]:
        """Perform search with retry logic and improved parsing"""
        driver = None
        try:
            driver = self.initialize_browser()

            # Clean and encode query for URL
            query = re.sub(r'[^\w\s-]', '', query.strip())
            query = re.sub(r'\s+', ' ', query).strip()
            encoded_query = '+'.join(query.split())
            url = f"https://duckduckgo.com/?q={encoded_query}&kl=us-en&k1=-1&atb=v233-1&ia=web"
            
            logger.info(f"Searching DuckDuckGo: {url}")
            driver.get(url)
            
            # Wait for results with better error handling
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="result"]'))
                )
                time.sleep(2)
            except TimeoutException:
                logger.warning("Timeout waiting for search results, retrying...")
                driver.refresh()
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="result"]'))
                )
                time.sleep(2)

            # JavaScript for result extraction
            js_script = """
            function getResults() {
                const results = [];
                const articles = document.querySelectorAll('article[data-testid="result"]');
                const maxResults = Math.min(articles.length, 8);
                
                for (let i = 0; i < maxResults; i++) {
                    try {
                        const article = articles[i];
                        const titleElem = article.querySelector('h2 a');
                        const snippetElem = article.querySelector('[data-result="snippet"] .kY2IgmnCmOGjharHErah');
                        const urlElem = article.querySelector('.pAgARfGNTRe_uaK72TAD a');
                        
                        if (titleElem && snippetElem && urlElem) {
                            const url = urlElem.href;
                            if (!url.includes('duckduckgo.com')) {
                                results.push({
                                    title: titleElem.textContent.trim(),
                                    url: url,
                                    snippet: snippetElem.textContent.trim()
                                });
                            }
                        }
                    } catch (e) {
                        console.error('Error parsing result:', e);
                    }
                }
                return results;
            }
            return getResults();
            """
            
            raw_results = driver.execute_script(js_script)
            
            # Process and validate results
            results = []
            for r in raw_results:
                result = SearchResult(**r)
                if result.is_valid():
                    results.append(result)
            
            if not results:
                logger.warning("No valid results found, may need to retry")
                raise ValueError("No valid results found")
                
            return results[:5]  # Return top 5 valid results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
            
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

class OllamaLLM:
    """Handles interactions with the Ollama phi4 model with improved prompting"""
    def __init__(self, model_name="phi4"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate(self, prompt: str) -> str:
        """Generate response with retry logic and improved error handling"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama API returned status {response.status_code}")
                    
                result = response.json()["response"].strip()
                if not result:
                    raise ValueError("Empty response from LLM")
                    
                return result
                
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            raise

@dataclass
class ExtractedContent:
    """Represents content extracted from a webpage"""
    url: str
    title: str
    raw_text: str
    relevant_snippets: List[str] = None

    def __post_init__(self):
        if self.relevant_snippets is None:
            self.relevant_snippets = []

class WebContentExtractor:
    """Handles webpage content extraction with proper cleanup"""
    def __init__(self):
        self.lock = threading.Lock()
        self._init_connection_pool()

    def _init_connection_pool(self):
        """Initialize connection pool settings"""
        import urllib3
        urllib3.PoolManager(maxsize=10, retries=urllib3.Retry(3))
        # Set Firefox capabilities for better connection handling
        self.firefox_capabilities = webdriver.FirefoxOptions()
        self.firefox_capabilities.add_argument('--headless')
        self.firefox_capabilities.add_argument('--disable-gpu')
        self.firefox_capabilities.add_argument('--no-sandbox')
        self.firefox_capabilities.add_argument('--disable-dev-shm-usage')
        self.firefox_capabilities.add_argument('--disable-javascript')
        self.firefox_capabilities.set_preference("permissions.default.stylesheet", 2)
        self.firefox_capabilities.set_preference("permissions.default.image", 2)
        # Add connection pooling settings
        self.firefox_capabilities.set_preference("network.http.connection-timeout", 10)
        self.firefox_capabilities.set_preference("network.http.max-connections", 10)
        self.firefox_capabilities.set_preference("network.http.max-persistent-connections-per-server", 5)

    def initialize_browser(self):
        """Initialize a new browser instance with proper configuration"""
        with self.lock:
            try:
                driver = webdriver.Firefox(options=self.firefox_capabilities)
                driver.set_page_load_timeout(15)
                driver.implicitly_wait(5)
                return driver
            except WebDriverException as e:
                logger.error(f"Failed to initialize Firefox: {e}")
                raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
    def extract_content(self, url: str) -> Optional[str]:
        """Extract and clean webpage content with improved connection handling"""
        driver = None
        try:
            driver = self.initialize_browser()
            driver.get(url)
            
            # Wait for content to load with more specific conditions
            WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body, p, article"))
            )
            # Add a small delay to ensure dynamic content loads
            time.sleep(2)

            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe']):
                element.decompose()

            # Extract text content
            text_content = ' '.join([p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
            
            # Clean up text
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = re.sub(r'[^\w\s.,!?-]', '', text_content)
            
            return text_content.strip()

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

class EnhancedLearningAgent:
    """Enhanced learning agent with content extraction capability"""
    def __init__(self):
        self.llm = OllamaLLM()
        self.search_engine = DuckDuckGoSearch()
        self.content_extractor = WebContentExtractor()

    async def generate_search_queries(self, question: str) -> List[str]:
        """Generate multiple optimized search queries to cover different aspects of the question"""
        prompt = f"""
        Break down this question into 2-3 focused search queries. Each query should:
        - Target a specific aspect of the question
        - Use 3-5 key terms only
        - Focus on authoritative historical sources
        - Avoid quotes or special characters
        
        Question: {question}
        
        Return ONLY the search queries, one per line.
        """
        
        try:
            response = await self.llm.generate(prompt)
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            logger.info(f"Generated search queries: {queries}")
            return queries[:3]  # Limit to maximum 3 queries
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return [question]  # Fallback to original question

    async def process_url(self, search_result: SearchResult, question: str) -> Optional[ExtractedContent]:
        """Process a single URL with content extraction"""
        try:
            raw_text = self.content_extractor.extract_content(search_result.url)
            if not raw_text:
                return None

            # Use existing LLM to analyze relevance
            analysis_prompt = f"""
            Analyze this text content and extract ONLY the parts that are directly relevant to answering this question:
            Question: {question}
            Content: {raw_text[:2000]}  # Limit content length
            
            Return only the relevant text snippets, separated by newlines.
            """
            
            relevant_text = await self.llm.generate(analysis_prompt)
            relevant_snippets = [s.strip() for s in relevant_text.split('\n') if s.strip()]

            return ExtractedContent(
                url=search_result.url,
                title=search_result.title,
                raw_text=raw_text,
                relevant_snippets=relevant_snippets
            )

        except Exception as e:
            logger.error(f"Error processing URL {search_result.url}: {e}")
            return None

    async def process_user_input(self, user_input: str) -> Dict:
        """Process user input with iterative searches and enhanced result synthesis"""
        try:
            # Generate search queries using original method
            search_queries = await self.generate_search_queries(user_input)
            
            all_results = []
            for query in search_queries:
                # Use original search method
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.search_engine.search, query)
                    search_results = future.result(timeout=30)
                    if search_results:
                        all_results.extend(search_results)
            
            if not all_results:
                return {
                    "status": "no_results",
                    "result": "I couldn't find reliable information for your question.",
                    "context": None
                }

            # Process URLs in parallel
            extracted_contents = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for result in all_results[:5]:  # Limit to top 5 results
                    future = executor.submit(
                        lambda r=result: asyncio.run(self.process_url(r, user_input))
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    content = future.result()
                    if content and content.relevant_snippets:
                        extracted_contents.append(content)

            # Generate final response using extracted content
            synthesis_prompt = f"""
            Question: {user_input}

            Using these verified sources, provide a precise answer that:
            1. Directly addresses all aspects of the question
            2. Uses only the provided content
            3. Cites sources with [n] format
            4. Maintains factual accuracy
            
            Sources:
            """

            for i, content in enumerate(extracted_contents, 1):
                synthesis_prompt += f"\n[{i}] URL: {content.url}\n"
                for snippet in content.relevant_snippets:
                    synthesis_prompt += f"Snippet: {snippet}\n"

            response = await self.llm.generate(synthesis_prompt)

            return {
                "status": "completed",
                "result": response,
                "context": [
                    {
                        "url": content.url,
                        "title": content.title,
                        "snippets": content.relevant_snippets
                    }
                    for content in extracted_contents
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {
                "status": "error",
                "result": f"An error occurred while processing your question. Please try again.",
                "context": None
            }

def format_response(result: Dict) -> str:
    """Format the response with improved readability"""
    output = []
    
    # Add status-specific headers
    if result["status"] == "completed":
        output.append("Answer:")
    elif result["status"] == "error":
        output.append("Error:")
    elif result["status"] == "no_results":
        output.append("No Results:")
    
    # Add the main response
    if result.get("result"):
        output.append(result["result"])
    
    # Add formatted references
    if result.get("context"):
        output.append("\nReferences:")
        for i, source in enumerate(result["context"], 1):
            # Clean up title and URL for display
            title = re.sub(r'\s+', ' ', source.get('title', 'No Title'))
            url = source.get('url', '').split('?')[0]  # Remove query parameters
            
            output.append(f"[{i}] {title}")
            output.append(f"    {url}")
    
    return "\n".join(output)

async def main():
    """Main interaction loop with enhanced agent"""
    print("\nEnhanced Learning Agent initialized. Enter your question (or 'quit' to exit):")
    agent = EnhancedLearningAgent()
    
    while True:
        try:
            user_input = input("\nYour question: ").strip()
            if user_input.lower() in ('quit', 'exit'):
                break
            if not user_input:
                continue
                
            print("\nProcessing your question...")
            result = await agent.process_user_input(user_input)
            print("\n" + format_response(result))
            
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
            print("Please try again or type 'quit' to exit.")
            continue

if __name__ == "__main__":
    asyncio.run(main())