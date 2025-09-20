import os
from typing import List, Dict
import PyPDF2
import docx

class DocumentLoader:
    """
    Loads documents from a local directory and splits them into chunks
    for embedding and retrieval.
    """

    def __init__(self, data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        :param data_dir: directory containing documents
        :param chunk_size: max number of characters per chunk
        :param chunk_overlap: number of overlapping characters between chunks
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _read_pdf(self, path: str) -> str:
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _read_docx(self, path: str) -> str:
        text = ""
        doc = docx.Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def _read_txt(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def load(self) -> List[Dict]:
        """
        Load all documents from data_dir, split into chunks, and return a list of dicts:
        {'id': <filename_chunkIndex>, 'text': <chunk_text>, 'meta': {'source': filename}}
        """
        docs = []
        for fname in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, fname)
            if fname.lower().endswith(".pdf"):
                text = self._read_pdf(path)
            elif fname.lower().endswith(".docx"):
                text = self._read_docx(path)
            elif fname.lower().endswith(".txt"):
                text = self._read_txt(path)
            else:
                continue  # skip unsupported file types

            if not text.strip():
                continue  # skip empty files

            chunks = self._chunk_text(text)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "id": f"{fname}_chunk{i}",
                    "text": chunk.strip(),
                    "meta": {"source": fname}
                })

        return docs
