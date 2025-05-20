# File: auto_browser.py

import asyncio
import base64
import os
import json
from playwright.async_api import async_playwright
from PIL import Image
import io
import requests
from dotenv import load_dotenv
import pytesseract
import cv2
import numpy as np


# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Default to OpenAI if not specified
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  

# Main class for our automation
class AutoBrowser:
    def __init__(self, headless=False):
        self.headless = headless
        self.browser = None
        self.context = None
        self.page = None
        self.action_history = []
        
    async def start(self):
        """Initialize the browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        print("Browser started")
        
    async def navigate(self, url):
        """Navigate to a URL"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        await self.page.goto(url)
        print(f"Navigated to {url}")
        
    async def close(self):
        """Close the browser"""
        if self.browser:
            await self.browser.close()
            print("Browser closed")

    # Add these methods to the AutoBrowser class
    async def capture_page_state(self):
        """Capture the current state of the page"""
        print("Capturing page state...")
        
        # Get page metadata
        url = self.page.url
        title = await self.page.title()
        
        # Take screenshot
        screenshot_bytes = await self.page.screenshot()
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        # Extract visible text
        text_content = await self.page.evaluate('''() => {
            return document.body.innerText;
        }''')
        
        # Extract interactive elements (simplified version)
        elements = await self.page.evaluate('''() => {
            const getVisibleElements = () => {
                // Get clickable elements
                const clickables = Array.from(document.querySelectorAll('button, a, input, select, [role="button"]'));
                
                return clickables.map(el => {
                    // Get text content
                    const text = el.innerText || el.value || el.placeholder || '';
                    
                    // Get element type
                    const tag = el.tagName.toLowerCase();
                    
                    // Get attributes
                    const id = el.id;
                    const classes = Array.from(el.classList).join(' ');
                    const type = el.type || '';
                    const href = el.href || '';
                    const role = el.getAttribute('role') || '';
                    
                    // Get position for visual reference
                    const rect = el.getBoundingClientRect();
                    const position = {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    };
                    
                    // Try to get a good description of the element
                    let description = text.trim();
                    if (!description) {
                        if (el.getAttribute('aria-label')) {
                            description = el.getAttribute('aria-label');
                        } else if (el.getAttribute('title')) {
                            description = el.getAttribute('title');
                        } else if (el.getAttribute('name')) {
                            description = el.getAttribute('name');
                        } else if (id) {
                            description = id;
                        }
                    }
                    
                    return {
                        tag,
                        text: text.trim(),
                        id,
                        classes,
                        type,
                        href,
                        role,
                        position,
                        description
                    };
                }).filter(el => {
                    // Filter out elements with no text and no description
                    return el.text || el.description;
                });
            };
            
            return getVisibleElements();
        }''')
        
        # Compile the full state
        page_state = {
            "url": url,
            "title": title,
            "text_content": text_content,
            "interactive_elements": elements,
            "screenshot_base64": screenshot_base64
        }
        
        return page_state

    async def wait_for_stable(self, timeout=5000):
        """Wait for the page to stabilize"""
        try:
            # Wait for network to be idle (no requests for 500ms)
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
            
            # Wait a bit more for any JavaScript animations
            await asyncio.sleep(1)
            
            print("Page stabilized")
            return True
        except Exception as e:
            print(f"Warning: Page did not stabilize completely: {e}")
            # Continue anyway
            return False  

    # Add to AutoBrowser class:
    async def _execute_search(self, search_query):
        """Execute a search on a search engine"""
        print(f"Searching for: {search_query}")
        
        # Detect which search engine we're on
        current_url = self.page.url.lower()
        
        if "google.com" in current_url:
            # For Google
            await self.page.fill('input[name="q"]', search_query)
            await self.page.press('input[name="q"]', 'Enter')
        elif "bing.com" in current_url:
            # For Bing
            await self.page.fill('input[name="q"]', search_query)
            await self.page.press('input[name="q"]', 'Enter')
        elif "duckduckgo.com" in current_url:
            # For DuckDuckGo
            await self.page.fill('input[name="q"]', search_query)
            await self.page.press('input[name="q"]', 'Enter')
        elif "yahoo.com" in current_url:
            # For Yahoo
            await self.page.fill('input[name="p"]', search_query)
            await self.page.press('input[name="p"]', 'Enter')
        else:
            # Generic approach - try common search input patterns
            try:
                # Try to find a search input
                search_input = await self.page.query_selector('input[type="search"], input[name="q"], input[name="search"], input[placeholder*="search" i]')
                if search_input:
                    await search_input.fill(search_query)
                    await search_input.press('Enter')
                    return True
            except Exception as e:
                print(f"Failed to execute search: {e}")
                return False
        
        return True
    
    # Add to AutoBrowser class:
    async def _find_and_click_result(self, site_name):
        """Find and click a search result matching the site name"""
        print(f"Looking for search result: {site_name}")
        
        # Simple approach - look for links containing the site name
        result = await self.page.query_selector(f'a[href*="{site_name}"]')
        if result:
            print(f"Found result for {site_name}, clicking...")
            await result.click()
            return True
        
        # More complex approach - evaluate all links and find the best match
        links = await self.page.evaluate('''(siteName) => {
            const links = Array.from(document.querySelectorAll('a'));
            return links.map(link => {
                return {
                    href: link.href,
                    text: link.innerText,
                    isVisible: link.offsetParent !== null
                };
            }).filter(link => link.isVisible && (
                link.href.includes(siteName) || 
                link.text.toLowerCase().includes(siteName.toLowerCase())
            ));
        }''', site_name)
        
        if links and len(links) > 0:
            # Click the first matching link
            best_link = links[0]
            print(f"Found result: {best_link['text']}")
            await self.page.goto(best_link['href'])
            return True
            
        print(f"No search result found for {site_name}")
        return False
    
    # Add these methods to the AutoBrowser class
    async def execute_action(self, decision):
        """Execute the action based on AI decision"""
        action_type = decision["action_type"]
        element_index = decision.get("element_index")
        input_value = decision.get("input_value")
        
        # Record this action in history
        self.action_history.append({
            "action_type": action_type,
            "element_index": element_index,
            "input_value": input_value,
            "description": decision["description"]
        })
        
        if action_type == "click":
            return await self._execute_click(element_index)
        elif action_type == "type":
            return await self._execute_type(element_index, input_value)
        elif action_type == "navigate":
            return await self.navigate(input_value)
        elif action_type == "search":
            return await self._execute_search(input_value)  # Add this line
        elif action_type == "wait":
            await asyncio.sleep(3)  # Simple wait
            return True
        elif action_type == "human_help" or action_type == "complete":
            # These are handled by the main loop
            return True
        else:
            print(f"Unknown action type: {action_type}")
            return False

    async def _execute_click(self, element_index):
        """Click on the specified element"""
        if element_index is None:
            print("Error: No element index provided for click action")
            return False
        
        # Get the current elements
        page_state = await self.capture_page_state()
        elements = page_state["interactive_elements"]
        
        if element_index < 0 or element_index >= len(elements):
            print(f"Error: Element index {element_index} out of range (0-{len(elements)-1})")
            return False
        
        element = elements[element_index]
        print(f"Clicking on: {element['description'] or element['text']}")
        
        # Calculate center point of the element
        x = element["position"]["x"] + element["position"]["width"] / 2
        y = element["position"]["y"] + element["position"]["height"] / 2
        
        # Click at the position
        await self.page.mouse.click(x, y)
        
        # Check for new tab/window
        await self._handle_new_tab()
        
        return True

    async def _execute_type(self, element_index, text):
        """Type text into the specified element"""
        if element_index is None or text is None:
            print("Error: Missing element index or text for type action")
            return False
        
        # Get the current elements
        page_state = await self.capture_page_state()
        elements = page_state["interactive_elements"]
        
        if element_index < 0 or element_index >= len(elements):
            print(f"Error: Element index {element_index} out of range (0-{len(elements)-1})")
            return False
        
        element = elements[element_index]
        print(f"Typing '{text}' into: {element['description'] or element['text']}")
        
        # Calculate center point of the element
        x = element["position"]["x"] + element["position"]["width"] / 2
        y = element["position"]["y"] + element["position"]["height"] / 2
        
        # Click at the position first
        await self.page.mouse.click(x, y)
        
        # Then type the text
        await self.page.keyboard.type(text)
        
        return True

    async def _handle_new_tab(self):
        """Handle any new tabs that might have opened"""
        pages = self.context.pages
        if len(pages) > 1 and pages[-1] != self.page:
            print("New tab detected, switching to it")
            self.page = pages[-1]
            await self.page.wait_for_load_state("domcontentloaded")
            return True
        return False              

# New website detection logic:
def extract_website_from_task(task):
    """Extract website mention from task description"""
    # Common travel booking sites
    travel_sites = {
        "makemytrip": "makemytrip.com",
        "cleartrip": "cleartrip.com",
        "goibibo": "goibibo.com",
        "expedia": "expedia.com",
        "booking": "booking.com",
        "kayak": "kayak.com",
        # "skyscanner": "skyscanner.com",
        # Airlines
        "indigo": "goindigo.in",
        "airasia": "airasia.com",
        "airindia": "airindia.in",
        "spicejet": "spicejet.com",
        "vistara": "airvistara.com",
        "emirates": "emirates.com",
        "lufthansa": "lufthansa.com",
        "britishairways": "britishairways.com",
        "jetblue": "jetblue.com",
        "united": "united.com",
        "delta": "delta.com",
        "klm": "klm.com",
        "easyjet": "easyjet.com",
        "ryanair": "ryanair.com",
        # Add more airlines and travel sites
    }
    
    # Check for explicit mentions of websites
    task_lower = task.lower()
    for site_keyword, site_url in travel_sites.items():
        if site_keyword in task_lower:
            return site_url
    
    # If no specific site mentioned but "flight" is in task,
    # either use a default or let the AI decide
    if "flight" in task_lower and not any(site in task_lower for site in travel_sites):
        # Option 1: Use a default flight booking site
        return "skyscanner.com"  # or any default you prefer
        
        # Option 2: Let the AI decide by going to a search engine first
        # return "google.com"
    
    # Fallback to search engine
    return "google.com"

# Add this function:
def extract_flight_parameters(task):
    """Extract flight parameters from the task description"""
    import re
    
    # Initialize parameters
    params = {
        "origin": None,
        "destination": None,
        "departure_date": None,
        "return_date": None,
        "passengers": 1,
        "class": "economy"
    }
    
    # Common Indian cities (for flight booking)
    indian_cities = [
        "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", 
        "kolkata", "pune", "jaipur", "ahmedabad", "kochi", "goa"
    ]
    
    # Common international cities
    international_cities = [
        "london", "new york", "dubai", "singapore", "bangkok", "paris",
        "tokyo", "sydney", "kuala lumpur", "hong kong", "toronto"
    ]
    
    # Combine all cities
    all_cities = indian_cities + international_cities
    
    # Look for city pairs
    task_lower = task.lower()
    
    # Pattern for "from X to Y"
    from_to_pattern = r'from\s+([a-zA-Z\s]+)\s+to\s+([a-zA-Z\s]+)'
    from_to_matches = re.findall(from_to_pattern, task_lower)
    
    if from_to_matches:
        origin_text, destination_text = from_to_matches[0]
        
        # Find best match for origin
        for city in all_cities:
            if city in origin_text:
                params["origin"] = city.title()
                break
                
        # Find best match for destination
        for city in all_cities:
            if city in destination_text:
                params["destination"] = city.title()
                break
    
    # Look for dates
    # Today, tomorrow, day after tomorrow
    if "today" in task_lower:
        from datetime import datetime
        params["departure_date"] = datetime.now().strftime("%Y-%m-%d")
    elif "tomorrow" in task_lower:
        from datetime import datetime, timedelta
        params["departure_date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "day after tomorrow" in task_lower:
        from datetime import datetime, timedelta
        params["departure_date"] = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    
    # Look for class
    if "business" in task_lower or "business class" in task_lower:
        params["class"] = "business"
    elif "first" in task_lower or "first class" in task_lower:
        params["class"] = "first"
    elif "premium" in task_lower or "premium economy" in task_lower:
        params["class"] = "premium economy"
    
    # Look for passengers
    passengers_pattern = r'(\d+)\s+passenger'
    passengers_matches = re.findall(passengers_pattern, task_lower)
    if passengers_matches:
        params["passengers"] = int(passengers_matches[0])
    
    return params

class AIDecisionEngine:
    def __init__(self, provider="anthropic"):
        self.provider = provider
        if provider == "anthropic":
            self.api_key = ANTHROPIC_API_KEY
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
            
    async def get_decision(self, task, page_state, action_history, task_params=None):
        """Get the next action decision from the AI"""
        return await self._get_anthropic_decision(task, page_state, action_history, task_params)
            
    async def _get_anthropic_decision(self, task, page_state, action_history, task_params=None):
        """Get decision using Anthropic's API with OCR fallback"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Optimize elements description to be more concise
        elements_text = ""
        for i, element in enumerate(page_state["interactive_elements"]):
            if element['text'] or element['description']:  # Only include elements with text
                elements_text += f"{i+1}. {element['description'] or element['text']} ({element['tag']})\n"
        
        # Optimize action history to be more concise
        history_text = "|".join([action['description'] for action in action_history[-3:]])  # Only last 3 actions

        # Optimize parameters text
        params_text = ""
        if task_params:
            params_text = "Params: " + ", ".join([f"{k}:{v}" for k, v in task_params.items() if v])

        # Extract text from screenshot using OCR
        ocr_text = self._extract_text_from_screenshot(page_state["screenshot_base64"])
        
        # Create the prompt with system instructions included
        prompt = f"""You are a web automation assistant. Decide the next action to complete the task.
        Return a JSON object with: action_type (click|type|navigate|wait|search|complete|human_help),
        element_index (null or int), input_value (null or str), description, reasoning, needs_human (bool),
        human_instruction (null or str), task_complete (bool).

        Task: {task}
        {params_text}
        URL: {page_state['url']}
        Text: {page_state['text_content'][:500]}...
        OCR: {ocr_text[:200]}...
        Elements: {elements_text}
        History: {history_text}"""
        
        try:
            # Call Anthropic's API with Haiku model
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,  # Reduced token limit
                system="You are a web automation assistant that helps with flight booking tasks. You analyze web pages and decide the next action to take. Always respond with a valid JSON object containing the action details.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            decision = json.loads(response.content[0].text)
            print(f"AI decision: {decision['action_type']}")
            
            return decision
            
        except Exception as e:
            print(f"Error with Anthropic API: {e}")
            # Fallback to OCR-based decision making
            return self._make_ocr_based_decision(task, page_state, action_history, task_params, ocr_text)
        
    def _extract_text_from_screenshot(self, screenshot_base64):
        """Extract text from screenshot using OCR with optimized processing"""
        try:
            # Convert base64 to image
            image_data = base64.b64decode(screenshot_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Optimize image size for faster processing
            height, width = image_cv.shape[:2]
            if width > 1000:
                scale = 1000 / width
                image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale)
            
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR with optimized config
            custom_config = r'--oem 3 --psm 6'  # Optimized for single uniform text block
            text = pytesseract.image_to_string(thresh, config=custom_config)
            return text
            
        except Exception as e:
            print(f"Error in OCR: {e}")
            return ""

    def _make_ocr_based_decision(self, task, page_state, action_history, task_params, ocr_text):
        """Make a decision based on OCR text when API fails"""
        # Simple rule-based decision making based on OCR text
        ocr_text_lower = ocr_text.lower()
        task_lower = task.lower()
        
        # Check for common elements with optimized matching
        if any(word in ocr_text_lower for word in ["search", "find"]) and "flight" in task_lower:
            return {
                "action_type": "search",
                "input_value": task,
                "description": "Searching for flights",
                "reasoning": "Found search box",
                "needs_human": False,
                "human_instruction": None,
                "task_complete": False
            }
            
        # Look for form fields with optimized matching
        for i, element in enumerate(page_state["interactive_elements"]):
            if element["type"] == "text" and element["text"]:
                text_lower = element["text"].lower()
                if "from" in text_lower and task_params.get("origin"):
                    return {
                        "action_type": "type",
                        "element_index": i,
                        "input_value": task_params["origin"],
                        "description": f"Enter origin: {task_params['origin']}",
                        "reasoning": "Found origin field",
                        "needs_human": False,
                        "human_instruction": None,
                        "task_complete": False
                    }
                elif "to" in text_lower and task_params.get("destination"):
                    return {
                        "action_type": "type",
                        "element_index": i,
                        "input_value": task_params["destination"],
                        "description": f"Enter destination: {task_params['destination']}",
                        "reasoning": "Found destination field",
                        "needs_human": False,
                        "human_instruction": None,
                        "task_complete": False
                    }
        
        # Default to human help if no clear action
        return {
            "action_type": "human_help",
            "element_index": None,
            "input_value": None,
            "description": "Need assistance",
            "reasoning": "No clear action found",
            "needs_human": True,
            "human_instruction": "Please help navigate",
            "task_complete": False
        }

 # Updated run_automation function

async def run_automation(task):
    """Main function to run the automation"""
    print(f"Starting automation task: {task}")
    
    try:
        # Initialize browser and AI
        auto_browser = AutoBrowser(headless=False)
        ai_engine = AIDecisionEngine(provider=AI_PROVIDER)
        
        await auto_browser.start()
        
        # Extract website from task
        website = extract_website_from_task(task)
        flight_params = extract_flight_parameters(task)
            
        # Navigate to the website
        await auto_browser.navigate(website)
        
        # Main automation loop
        max_steps = 20  # Prevent infinite loops during development
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n--- Step {step_count} ---")
            
            try:
                # Wait for page to stabilize with increased timeout
                await auto_browser.wait_for_stable(timeout=10000)  # Increased to 10 seconds
                
                # Capture the current state of the page
                page_state = await auto_browser.capture_page_state()
                
                # Get AI decision
                decision = await ai_engine.get_decision(
                    task, 
                    page_state, 
                    auto_browser.action_history,
                    flight_params
                )
                
                # Check if human intervention is needed
                if decision["needs_human"]:
                    print("\n*** HUMAN INTERVENTION NEEDED ***")
                    print(decision["human_instruction"])
                    print("\nPress Enter when you've completed the requested action...")
                    input()
                    continue
                    
                # Check if task is complete
                if decision["task_complete"]:
                    print("\n*** TASK COMPLETE ***")
                    print(decision["description"])
                    break
                    
                # Execute the action
                success = await auto_browser.execute_action(decision)
                if not success:
                    print(f"Failed to execute action: {decision['action_type']}")
                    print("Press Enter to try again or Ctrl+C to exit...")
                    input()
                    
            except Exception as e:
                print(f"Error in automation step: {e}")
                print("Press Enter to continue or Ctrl+C to exit...")
                input()
                continue
    
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        # Ensure browser is closed properly
        try:
            await auto_browser.close()
        except Exception as e:
            print(f"Error closing browser: {e}")
        
        print("\nAutomation finished. Press Enter to exit...")
        input()

# Entry point
if __name__ == "__main__":
    print("Welcome to the Flight Booking Automation System!")
    print("\nPlease enter your flight booking task. For example:")
    print("'Book a flight from Bangalore to Mumbai today on MakeMyTrip'")
    print("'Find flights from London to Paris tomorrow on British Airways'")
    print("\nYour task: ", end="")
    
    task = input().strip()
    
    if not task:
        print("No task provided. Exiting...")
        exit(1)
        
    print(f"\nStarting automation for task: {task}")
    print("Initializing browser and AI engine...")
    
    # Run the automation with proper error handling
    try:
        asyncio.run(run_automation(task))
    except KeyboardInterrupt:
        print("\nAutomation interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        print("Script finished.")

if __name__ == "__main__":
    print("Welcome to the Flight Booking Automation System!")
    print("\nPlease enter your flight booking task. For example:")
    print("'Book a flight from Bangalore to Mumbai today on MakeMyTrip'")
    print("'Find flights from London to Paris tomorrow on British Airways'")
    print("\nYour task: ", end="")
    
    task = input().strip()
    
    if not task:
        print("No task provided. Exiting...")
        exit(1)
        
    print(f"\nStarting automation for task: {task}")
    print("Initializing browser and AI engine...")
    
    # Run the automation
    asyncio.run(run_automation(task))