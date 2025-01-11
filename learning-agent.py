import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
import time
import re
from concurrent.futures import ThreadPoolExecutor
import httpx
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        # Remove extra whitespace and normalize quotes
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('"', '"').replace('"', '"')
        return text

    def _clean_url(self, url: str) -> str:
        """Clean and validate URLs"""
        if not url:
            return ""
        # Remove tracking parameters and clean URL
        url = url.split('?')[0]
        # Remove common tracking endpoints
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
        self.driver = None
        self.max_retries = max_retries

    def initialize_browser(self):
        """Initialize browser with improved error handling"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None

        options = webdriver.FirefoxOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
        
        try:
            self.driver = webdriver.Firefox(options=options)
            self.driver.set_page_load_timeout(20)
            self.driver.implicitly_wait(5)
        except WebDriverException as e:
            logger.error(f"Failed to initialize Firefox: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def search(self, query: str) -> List[SearchResult]:
        """Perform search with retry logic and improved parsing"""
        try:
            if not self.driver:
                self.initialize_browser()

            # Clean and encode query for URL
            query = re.sub(r'[^\w\s-]', '', query.strip())  # Remove special chars
            query = re.sub(r'\s+', ' ', query).strip()
            encoded_query = '+'.join(query.split())
            url = f"https://duckduckgo.com/?q={encoded_query}&kl=us-en&k1=-1&atb=v233-1&ia=web"
            
            logger.info(f"Searching DuckDuckGo: {url}")
            self.driver.get(url)
            
            # Wait for results with better error handling and specific selectors
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="result"]'))
                )
                # Add a small delay to ensure dynamic content loads
                time.sleep(2)
            except TimeoutException:
                logger.warning("Timeout waiting for search results, retrying...")
                self.driver.refresh()
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="result"]'))
                )
                time.sleep(2)

            # Improved JavaScript for result extraction with new selectors
            js_script = """
            function getResults() {
                const results = [];
                const articles = document.querySelectorAll('article[data-testid="result"]');
                
                for (let i = 0; i < Math.min(articles.length, 8); i++) {
                    try {
                        const article = articles[i];
                        const titleElem = article.querySelector('h2 a');
                        const snippetElem = article.querySelector('[data-result="snippet"] .kY2IgmnCmOGjharHErah');
                        const urlElem = article.querySelector('.pAgARfGNTRe_uaK72TAD a');
                        
                        if (titleElem && snippetElem && urlElem) {
                            const url = urlElem.href;
                            // Skip if it's a DuckDuckGo internal link
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
            
            raw_results = self.driver.execute_script(js_script)
            
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
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = None

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

class LearningAgent:
    """Main learning agent class with improved coordination and error handling"""
    def __init__(self):
        self.llm = OllamaLLM()
        self.search_engine = DuckDuckGoSearch()

    async def generate_search_queries(self, question: str) -> List[str]:
        """Generate multiple optimized search queries to cover different aspects of the question"""
        prompt = f"""
        Break down this question into 2-3 focused search queries. Each query should:
        - Target a specific aspect of the question
        - Use 3-5 key terms only
        - Include the current year
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

    async def process_user_input(self, user_input: str) -> Dict:
        """Process user input with iterative searches and enhanced result synthesis"""
        task = Task(
            id=str(time.time()),
            description=user_input
        )
        
        try:
            logger.info(f"Processing query: {task.description}")
            
            # Generate multiple search queries
            search_queries = await self.generate_search_queries(task.description)
            
            all_results = []
            for query in search_queries:
                logger.info(f"Executing search for query: {query}")
                # Perform search with timeout
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.search_engine.search, query)
                    search_results = future.result(timeout=30)
                    if search_results:
                        all_results.extend(search_results)
                        logger.info(f"Found {len(search_results)} results for query: {query}")
            
            # Remove duplicates based on URL
            seen_urls = set()
            unique_results = []
            for result in all_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_results.append(result)
            
            if not unique_results:
                return {
                    "status": "no_results",
                    "result": "I couldn't find reliable information for your question. Could you please rephrase it or provide more context?",
                    "context": None
                }

            logger.info(f"Total unique results found: {len(unique_results)}")

            # Generate comprehensive response using all gathered information
            prompt = f"""
            Question: {task.description}

            Using these combined sources, provide a comprehensive answer that:
            1. Addresses all aspects of the question
            2. Uses [n] citations for facts and claims
            3. Synthesizes information from multiple sources when relevant
            4. Maintains natural, flowing language
            5. Organizes information logically
            6. Includes specific details, numbers, and dates when available
            7. Ensures each major claim has proper citation
            
            Sources:
            """
            
            for i, result in enumerate(unique_results, 1):
                prompt += f"\n[{i}] {result.title}"
                prompt += f"\nURL: {result.url}"
                prompt += f"\nContent: {result.snippet}\n"
            
            response = await self.llm.generate(prompt)
            
            # Format the result
            result = {
                "status": "completed",
                "result": response,
                "context": [r.to_dict() for r in unique_results]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return {
                "status": "error",
                "result": f"An error occurred while processing your question. Please try again.",
                "context": None
            }

    async def optimize_query(self, question: str) -> str:
        """Use LLM to optimize the search query with improved prompting"""
        prompt = f"""
        Transform this question into a simple search query to find accurate, recent information.
        Guidelines:
        - Use 3-5 key terms only
        - Include the year for time-sensitive info
        - NO quotes, commas, or special characters
        - Keep it simple and direct
        
        Question: {question}
        
        Return ONLY the optimized search terms, no punctuation or explanations.
        """
        
        try:
            optimized = await self.llm.generate(prompt)
            # Clean up the query - remove quotes, commas and excessive spaces
            optimized = re.sub(r'["\',]', '', optimized.strip())
            optimized = re.sub(r'\s+', ' ', optimized)
            
            logger.info(f"Original query: {question}")
            logger.info(f"Optimized query: {optimized}")
            
            return optimized
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return question

    async def process_user_input(self, user_input: str) -> Dict:
        """Process user input with iterative searches and enhanced result synthesis"""
        task = Task(
            id=str(time.time()),
            description=user_input
        )
        
        try:
            # Generate multiple search queries
            search_queries = await self.generate_search_queries(task.description)
            logger.info(f"Generated queries: {search_queries}")
            
            all_results = []
            for query in search_queries:
                # Perform search with timeout
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.search_engine.search, query)
                    search_results = future.result(timeout=30)
                    all_results.extend(search_results)
            
            if not all_results:
                return {
                    "status": "no_results",
                    "result": "I couldn't find reliable information for your question. Could you please rephrase it or provide more context?",
                    "context": None
                }

            # Generate comprehensive response using all gathered information
            prompt = f"""
            Question: {task.description}

            Using these combined sources, provide a comprehensive answer that:
            1. Addresses all aspects of the question
            2. Uses [n] citations for facts and claims
            3. Synthesizes information from multiple sources when relevant
            4. Maintains natural, flowing language
            5. Organizes information logically
            6. Includes specific details, numbers, and dates when available
            7. Ensures each major claim has proper citation
            
            Sources:
            """
            
            for i, result in enumerate(all_results, 1):
                prompt += f"\n[{i}] {result.title}"
                prompt += f"\nURL: {result.url}"
                prompt += f"\nContent: {result.snippet}\n"
            
            response = await self.llm.generate(prompt)
            
            # Format the result
            result = {
                "status": "completed",
                "result": response,
                "context": [r.to_dict() for r in all_results]
            }
            
            return result
            
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
    if result["result"]:
        output.append(result["result"])
    
    # Add formatted references
    if result["context"]:
        output.append("\nReferences:")
        for i, source in enumerate(result["context"], 1):
            # Clean up title and URL for display
            title = re.sub(r'\s+', ' ', source['title'])
            url = source['url'].split('?')[0]  # Remove query parameters
            
            output.append(f"[{i}] {title}")
            output.append(f"    {url}")
    
    return "\n".join(output)

async def main():
    """Main interaction loop with improved error handling"""
    print("\nLearning Agent initialized. Enter your question (or 'quit' to exit):")
    agent = LearningAgent()
    
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
    import asyncio
    asyncio.run(main())