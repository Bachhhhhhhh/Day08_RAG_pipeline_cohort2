"""
Task 1 - Thu thap van ban phap luat ve ma tuy va cac chat cam.

Du lieu duoc tai tu Cong Thong tin dien tu Chinh phu. Script doc trang chi
tiet van ban, tim tep PDF dinh kem va luu tep goc vao data/landing/legal/.
"""

import re
from pathlib import Path
from urllib.parse import urljoin

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "page_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=204940",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=204678",
        "filename": "nghi-dinh-105-2021.pdf",
    },
    {
        "page_url": "https://vanban.chinhphu.vn/?pageid=27160&docid=206454",
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf",
    },
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/125.0 Safari/537.36"
    )
}


def setup_directory() -> None:
    """Tao thu muc data/landing/legal/ neu chua co."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def find_pdf_attachment(page_url: str) -> str:
    """Tra ve URL cua tep PDF dinh kem dau tien tren trang van ban."""
    response = requests.get(
        page_url, headers=REQUEST_HEADERS, timeout=30
    )
    response.raise_for_status()

    hrefs = re.findall(
        r"""href\s*=\s*["']([^"']+\.pdf(?:\?[^"']*)?)["']""",
        response.text,
        flags=re.IGNORECASE,
    )
    if not hrefs:
        raise ValueError(f"Khong tim thay tep PDF dinh kem tai {page_url}")
    return urljoin(page_url, hrefs[0])


def download_file(url: str, filename: str) -> Path:
    """Tai mot tep PDF, kiem tra dinh dang va luu vao DATA_DIR."""
    output_path = DATA_DIR / filename
    response = requests.get(
        url, headers=REQUEST_HEADERS, timeout=60
    )
    response.raise_for_status()

    content = response.content
    if len(content) <= 1024 or not content.startswith(b"%PDF"):
        raise ValueError(f"Du lieu tai ve khong phai PDF hop le: {url}")

    output_path.write_bytes(content)
    return output_path


def collect_legal_documents() -> list[Path]:
    """Tai tat ca van ban phap luat duoc cau hinh."""
    setup_directory()
    downloaded = []

    for document in LEGAL_DOCUMENTS:
        attachment_url = find_pdf_attachment(document["page_url"])
        output_path = download_file(attachment_url, document["filename"])
        downloaded.append(output_path)
        print(f"Downloaded: {output_path.name}")

    return downloaded


if __name__ == "__main__":
    collect_legal_documents()
