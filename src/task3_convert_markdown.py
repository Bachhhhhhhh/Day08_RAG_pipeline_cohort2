"""
Task 3 - Convert toan bo du lieu landing sang Markdown.

PDF/DOC/DOCX duoc convert bang MarkItDown. JSON tu Task 2 da chua Markdown,
nen module tao metadata header va giu nguyen noi dung da crawl.
"""

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_METADATA = {
    "luat-phong-chong-ma-tuy-2021": {
        "title": "Luat Phong, chong ma tuy 2021",
        "source": "https://vanban.chinhphu.vn/?pageid=27160&docid=204940",
    },
    "nghi-dinh-105-2021": {
        "title": "Nghi dinh 105/2021/ND-CP",
        "source": "https://vanban.chinhphu.vn/?pageid=27160&docid=204678",
    },
    "nghi-dinh-57-2022-danh-muc-chat-ma-tuy": {
        "title": "Nghi dinh 57/2022/ND-CP",
        "source": "https://vanban.chinhphu.vn/?pageid=27160&docid=206454",
    },
}


def convert_legal_docs() -> list[Path]:
    """Convert cac van ban phap luat sang Markdown."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = MarkItDown()
    converted = []
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        result = converter.convert(str(filepath))
        content = result.text_content.strip()
        if not content:
            metadata = LEGAL_METADATA.get(filepath.stem, {})
            content = (
                f"# {metadata.get('title', filepath.stem)}\n\n"
                f"**Source:** {metadata.get('source', 'N/A')}\n\n"
                f"**Original file:** {filepath.name}\n\n"
                "Tai lieu goc la ban PDF scan khong co text layer. MarkItDown "
                "khong the trich xuat noi dung chu ma khong dung them OCR. "
                "Ban PDF goc van duoc giu trong data/landing/legal de doi "
                "chieu va xu ly OCR o giai doan sau."
            )

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content + "\n", encoding="utf-8")
        converted.append(output_path)
        print(f"Converted: {filepath.name} -> {output_path.name}")

    return converted


def convert_news_articles() -> list[Path]:
    """Chuyen cac bai bao JSON thanh Markdown co metadata header."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for filepath in sorted(news_dir.glob("*.json")):
        data = json.loads(filepath.read_text(encoding="utf-8"))
        body = (
            data.get("content_markdown")
            or data.get("content")
            or ""
        ).strip()
        if not body:
            raise ValueError(f"{filepath.name} khong co noi dung bai bao")

        title = data.get("title") or "Unknown"
        header = (
            f"# {title}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n"
            "---\n\n"
        )
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + body + "\n", encoding="utf-8")
        converted.append(output_path)
        print(f"Converted: {filepath.name} -> {output_path.name}")

    return converted


def convert_all() -> list[Path]:
    """Convert tat ca tep hop le va giu cau truc legal/news."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return convert_legal_docs() + convert_news_articles()


if __name__ == "__main__":
    files = convert_all()
    print(f"Done: {len(files)} Markdown files in {OUTPUT_DIR}")
