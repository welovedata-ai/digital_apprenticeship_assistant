"""
Scraper for Skills England apprenticeship pages.

Uses Playwright (headless Chromium) to handle JavaScript-rendered content,
since the Skills England site requires JS execution to display page content.

Ethical scraping principles applied:
  - Fixed URL list only — no auto-crawling or link following.
  - Descriptive User-Agent string identifying this as a student research project.
  - Configurable delay between each request (default: 2 seconds).
  - Pages are saved once to disk; re-running skips already-saved files.

Run with:
    python -m src.scraper
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright

from src.config import STANDARD_URLS, RAW_DIR, SCRAPE_DELAY_SECONDS, USER_AGENT


def url_to_filename(url: str) -> str:
    """Convert a URL into a safe JSON filename using the last path segment."""
    slug = url.rstrip("/").split("/")[-1]
    return f"{slug}.json"


def detect_page_type(url: str) -> str:
    """Infer the page type from the URL path."""
    if "/apprenticeship-units/" in url:
        return "apprenticeship_unit"
    if "/apprenticeships/" in url:
        return "apprenticeship_standard"
    return "unknown"


async def scrape_page(browser, url: str) -> dict:
    """
    Open a single URL in a new browser tab and extract its content.

    Returns a dict containing:
        url, title, scraped_at, status_code, page_type,
        html_content, text_content, error
    """
    page = await browser.new_page(
        extra_http_headers={"User-Agent": USER_AGENT}
    )

    result = {
        "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "title": "",
        "status_code": None,
        "page_type": detect_page_type(url),
        "html_content": "",
        "text_content": "",
        "error": None,
    }

    try:
        response = await page.goto(url, wait_until="networkidle", timeout=45_000)
        result["status_code"] = response.status if response else None

        await page.wait_for_load_state("domcontentloaded")

        result["title"] = await page.title()
        result["html_content"] = await page.content()

        # Extract readable text by removing non-content elements before
        # calling innerText — this avoids capturing hidden/script text.
        result["text_content"] = await page.evaluate("""() => {
            const remove = ['script', 'style', 'noscript', 'link', 'meta'];
            remove.forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove());
            });
            return document.body ? document.body.innerText : '';
        }""")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"    ✗ Error scraping {url}: {exc}")

    finally:
        await page.close()

    return result


async def scrape_all(skip_existing: bool = True) -> None:
    """
    Scrape every URL in STANDARD_URLS and save one JSON file per page
    to data/raw/.

    Args:
        skip_existing: If True (default), URLs whose output file already
                       exists are skipped, saving time on re-runs.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"StandardsBot Scraper — {len(STANDARD_URLS)} pages to scrape")
    print(f"Output directory: {RAW_DIR.resolve()}\n")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)

        for index, url in enumerate(STANDARD_URLS, start=1):
            filename = url_to_filename(url)
            output_path = RAW_DIR / filename

            if skip_existing and output_path.exists():
                print(f"[{index:>2}/{len(STANDARD_URLS)}] Skipping (already saved): {filename}")
                continue

            print(f"[{index:>2}/{len(STANDARD_URLS)}] Scraping: {url}")

            data = await scrape_page(browser, url)

            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)

            status = data["status_code"] or "error"
            chars = len(data["text_content"])
            print(f"    ✓ Saved {filename} — HTTP {status}, {chars:,} text chars")

            # Ethical delay between requests (skip after the final page)
            if index < len(STANDARD_URLS):
                print(f"    ⏳ Waiting {SCRAPE_DELAY_SECONDS}s before next request…")
                await asyncio.sleep(SCRAPE_DELAY_SECONDS)

        await browser.close()

    saved = len(list(RAW_DIR.glob("*.json")))
    print(f"\nDone. {saved} JSON file(s) in {RAW_DIR}/")


if __name__ == "__main__":
    asyncio.run(scrape_all())
