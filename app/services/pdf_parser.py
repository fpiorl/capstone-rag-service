from io import BytesIO

from pypdf import PdfReader


class PdfParser:
    def extract_text(self, pdf_bytes: bytes) -> str:
        if not pdf_bytes:
            return ""

        reader = PdfReader(BytesIO(pdf_bytes))
        texts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                texts.append(page_text.strip())

        return "\n\n".join(texts).strip()
