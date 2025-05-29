import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig


async def main():
    browser_config = BrowserConfig(browser_type="chromium",text_mode=True)  # Default browser configuration
    crawl_conf = CrawlerRunConfig(
        # js_code="document.querySelector('button#loadMore')?.click()",
        # wait_for="css:.loaded-content",
        scroll_delay=True,
        simulate_user=True
    )


    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.sciencedirect.com/science/article/abs/pii/S2352492823020706",
            config=crawl_conf
        )
        if result.success:
            print(result.markdown)  # Print clean markdown content
        else:
            print(f"Crawl failed with status code {result.status_code}")


    # Different content formats
    print(result.html)         # Raw HTML
    print(result.cleaned_html) # Cleaned HTML
    print(result.markdown)     # Markdown version
    print(result.fit_markdown) # Most relevant content in markdown

    # Check success status
    print(result.success)      # True if crawl succeeded
    print(result.status_code)  # HTTP status code (e.g., 200, 404)

    # Access extracted media and links
    print(result.media)        # Dictionary of found media (images, videos, audio)
    print(result.links)        # Dictionary of internal and external links

if __name__ == "__main__":
    asyncio.run(main())


