"""
Task 2 - Crawl bai bao ve nghe si Viet Nam lien quan toi ma tuy.

Moi bai bao duoc luu thanh mot tep JSON gom URL, tieu de, thoi diem crawl va
noi dung Markdown do Crawl4AI trich xuat.
"""

import asyncio
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://vnexpress.net/dien-vien-hai-huu-tin-bi-cao-buoc-to-chuc-choi-ma-tuy-4477400.html",
    "https://tuoitre.vn/dien-vien-huu-tin-bi-truy-to-vi-to-chuc-su-dung-ma-tuy-20221117104908287.htm",
    "https://vnexpress.net/hiep-ga-va-cuoc-song-khon-kho-vi-ma-tuy-1892564.html",
    "https://vnexpress.net/dien-vien-hai-huu-tin-su-dung-ma-tuy-vi-to-mo-4599355.html",
]


def setup_directory() -> None:
    """Tao thu muc data/landing/news/ neu chua co."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _markdown_text(result) -> str:
    """Ho tro ca kieu tra ve Markdown cu va moi cua Crawl4AI."""
    markdown = getattr(result, "markdown", "")
    if isinstance(markdown, str):
        return markdown
    return (
        getattr(markdown, "fit_markdown", None)
        or getattr(markdown, "raw_markdown", None)
        or str(markdown or "")
    )


def _crawl_with_requests(url: str) -> dict:
    """Fallback nhe khi browser cua Crawl4AI chua san sang."""
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/125.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    page = response.text

    title_match = re.search(
        r"<title[^>]*>(.*?)</title>",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    title = (
        html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()
        if title_match
        else "Unknown"
    )

    article_match = re.search(
        r"<article\b[^>]*>(.*?)</article>",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    article_html = article_match.group(1) if article_match else page
    article_html = re.sub(
        r"<(script|style|svg|nav|footer)\b.*?</\1>",
        "",
        article_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    article_html = re.sub(
        r"<(br|p|div|h[1-6]|li)\b[^>]*>",
        "\n",
        article_html,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"<[^>]+>", " ", article_html)
    content = html.unescape(content)
    content = "\n".join(
        re.sub(r"\s+", " ", line).strip()
        for line in content.splitlines()
        if re.sub(r"\s+", " ", line).strip()
    )
    if len(content) < 200:
        raise ValueError(f"Noi dung crawl qua ngan tai {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": f"# {title}\n\n{content}",
        "crawler": "requests_fallback",
    }


async def crawl_article(url: str, crawler=None) -> dict:
    """Crawl mot bai bao va tra ve metadata cung noi dung Markdown."""
    from crawl4ai import AsyncWebCrawler

    owns_crawler = crawler is None
    if owns_crawler:
        crawler = AsyncWebCrawler()
        await crawler.__aenter__()

    try:
        result = await crawler.arun(url=url)
        if hasattr(result, "success") and not result.success:
            error = getattr(result, "error_message", "unknown error")
            raise RuntimeError(f"Crawl that bai {url}: {error}")

        metadata = getattr(result, "metadata", None) or {}
        content = _markdown_text(result).strip()
        if len(content) < 200:
            raise ValueError(f"Noi dung crawl qua ngan tai {url}")

        return {
            "url": url,
            "title": metadata.get("title") or "Unknown",
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
        }
    finally:
        if owns_crawler:
            await crawler.__aexit__(None, None, None)


async def crawl_all() -> list[Path]:
    """Crawl toan bo ARTICLE_URLS bang mot browser session."""
    setup_directory()
    saved_files = []

    try:
        from crawl4ai import AsyncWebCrawler
    except ImportError:
        AsyncWebCrawler = None

    if AsyncWebCrawler is None:
        for index, url in enumerate(ARTICLE_URLS, start=1):
            print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
            article = await asyncio.to_thread(_crawl_with_requests, url)
            output_path = DATA_DIR / f"article_{index:02d}.json"
            output_path.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_files.append(output_path)
            print(f"Saved: {output_path.name}")
        return saved_files

    async with AsyncWebCrawler() as crawler:
        for index, url in enumerate(ARTICLE_URLS, start=1):
            print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
            article = await crawl_article(url, crawler=crawler)
            output_path = DATA_DIR / f"article_{index:02d}.json"
            output_path.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_files.append(output_path)
            print(f"Saved: {output_path.name}")

    return saved_files


if __name__ == "__main__":
    asyncio.run(crawl_all())
