import logging
from typing import Dict, Any, List
from pipeline.config import get_settings

try:
    from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext, TimeoutError as PlaywrightTimeoutError
except ImportError:
    sync_playwright = None
    Page = None
    Browser = None
    BrowserContext = None
    PlaywrightTimeoutError = Exception

logger = logging.getLogger(__name__)

# A simple JavaScript snippet to extract interactive elements and assign them IDs
_SNAPSHOT_JS = """
() => {
    let elementId = 1;
    const elements = [];
    const interactiveSelectors = 'a, button, input, select, textarea, [role="button"], [role="link"], [tabindex]:not([tabindex="-1"])';
    
    function walk(root) {
        if (!root) return;
        
        if (root.nodeType === 1) { // Node.ELEMENT_NODE
            const el = root;
            if (el.matches && el.matches(interactiveSelectors)) {
                const style = window.getComputedStyle(el);
                if (style.display !== 'none' && style.visibility !== 'hidden' && el.offsetWidth > 0 && el.offsetHeight > 0) {
                    let tag = el.tagName.toLowerCase();
                    let text = el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || el.title || '';
                    if (!text.trim()) {
                        text = el.id || (el.className && typeof el.className === 'string' ? el.className : '');
                    }
                    text = text.trim().substring(0, 100).replace(/\\n/g, ' ');
                    
                    if (tag === 'input') {
                        const type = el.getAttribute('type') || 'text';
                        tag = `input[type="${type}"]`;
                    }
                    
                    let actionHint = 'CLICK ONLY';
                    if (tag.startsWith('input') || tag === 'textarea') {
                        actionHint = 'TYPE HERE';
                    }
                    
                    if (text || tag.startsWith('input') || tag === 'button') {
                        const id = elementId++;
                        el.setAttribute('data-browser-id', id);
                        elements.push(`[${id}] [${actionHint}] ${tag}: "${text}"`);
                    }
                }
            }
            
            if (el.shadowRoot) {
                walk(el.shadowRoot);
            }
        }
        
        const childNodes = root.childNodes || [];
        for (let i = 0; i < childNodes.length; i++) {
            walk(childNodes[i]);
        }
    }
    
    walk(document.body);
    return elements.join('\\n');
}
"""

class AntiBotException(Exception):
    """Raised when severe anti-bot protection is detected."""


class BrowserClient:
    def __init__(self):
        if sync_playwright is None:
            raise ImportError("playwright is not installed. Run `pip install playwright` and `playwright install chromium`.")
        self.playwright = sync_playwright().start()
        headless = get_settings().browser_headless
        self.browser = self.playwright.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        self.context = self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080}
        )
        # Add init script to mask webdriver
        self.context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        self.page = self.context.new_page()
        
    def _check_anti_bot(self):
        """Check if we've hit a Cloudflare or similar anti-bot wall."""
        try:
            title = self.page.title().lower()
            text = self.page.locator("body").inner_text().lower()
            if "cloudflare" in title or "just a moment..." in title or "attention required" in title:
                raise AntiBotException("Cloudflare Turnstile or similar anti-bot detected.")
            if "verify you are human" in text or "are you a robot" in text:
                raise AntiBotException("Captcha or human verification wall detected.")
        except PlaywrightTimeoutError:
            pass # Ignore timeouts during checks

    def navigate(self, url: str) -> bool:
        """Navigate to a URL and wait for it to load."""
        try:
            logger.info("Navigating to %s", url)
            self.page.goto(url, wait_until="load", timeout=30000)
            try:
                self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            self.page.wait_for_timeout(3000) # give JS time to settle
            self._check_anti_bot()
            return True
        except AntiBotException:
            raise
        except Exception as e:
            logger.error("Navigation failed: %s", e)
            return False

    def get_snapshot(self) -> str:
        """Returns a string representation of interactive elements on the page."""
        try:
            snapshot = self.page.evaluate(_SNAPSHOT_JS)
            return snapshot
        except Exception as e:
            logger.error("Snapshot failed: %s", e)
            return ""

    def get_html(self) -> str:
        """Returns the full HTML of the page."""
        try:
            return self.page.content()
        except Exception as e:
            logger.error("Failed to get HTML: %s", e)
            return ""

    def execute_actions(self, actions: List[Dict[str, Any]]) -> tuple[bool, str]:
        """Execute a list of sequential actions."""
        for action in actions:
            success, err = self._execute_single_action(action)
            if not success:
                return False, err
        return True, ""

    def _execute_single_action(self, action: Dict[str, Any]) -> tuple[bool, str]:
        """Execute a single browser action based on element_id."""
        action_type = action.get("action")
        element_id = action.get("element_id")
        text = action.get("text", "")

        if action_type == "done":
            return True, ""

        if not element_id:
            logger.warning("Action %s missing element_id", action_type)
            return False, "missing element_id"

        selector = f'[data-browser-id="{element_id}"]'
        
        try:
            logger.info("Executing action: %s on element %s", action_type, element_id)
            self.page.wait_for_selector(selector, state="attached", timeout=5000)
            element = self.page.locator(selector)
            
            element.scroll_into_view_if_needed()
            
            if action_type == "click":
                try:
                    element.click(timeout=5000)
                except PlaywrightTimeoutError:
                    logger.warning("Normal click timed out for %s, trying force click.", element_id)
                    try:
                        element.click(timeout=5000, force=True)
                    except Exception as e:
                        logger.warning("Force click failed (%s), trying JS click.", e)
                        element.evaluate("node => node.click()")
                except Exception as e:
                    logger.warning("Normal click failed with %s, trying JS click.", e)
                    element.evaluate("node => node.click()")
                # Wait for any potential navigation or network activity to settle
                try:
                    self.page.wait_for_load_state("load", timeout=10000)
                except Exception:
                    pass
                try:
                    self.page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                self.page.wait_for_timeout(3000)
            elif action_type == "type":
                try:
                    tag_name = element.evaluate("node => node.tagName.toLowerCase()")
                    if tag_name not in ['input', 'textarea']:
                        err = f"Element is a '{tag_name}', not an input or textarea. You MUST use 'click' on this element, not 'type'."
                        logger.warning(err)
                        return False, err
                    element.fill(text, timeout=5000)
                    self.page.wait_for_timeout(500)
                except Exception as e:
                    return False, f"Failed to type: {e}"
            else:
                err = f"Unknown action type: {action_type}"
                logger.warning(err)
                return False, err
                
            self._check_anti_bot()
            return True, ""
        except AntiBotException:
            raise
        except PlaywrightTimeoutError:
            err = f"Timeout executing action {action_type} on {element_id}."
            logger.error(err)
            return False, err
        except Exception as e:
            err = f"Failed to execute action {action_type} on {element_id}: {e}"
            logger.error(err)
            return False, err

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        try:
            self.context.close()
            self.browser.close()
            self.playwright.stop()
        except Exception:
            pass
