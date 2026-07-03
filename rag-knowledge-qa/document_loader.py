"""
文档加载器：支持 PDF / TXT / Markdown / Word
"""
import os
from pathlib import Path


def load_pdf(file_path: str) -> str:
    """从 PDF 提取纯文本"""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def load_docx(file_path: str) -> str:
    """从 Word 文档提取纯文本"""
    from docx import Document
    doc = Document(file_path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_txt(file_path: str) -> str:
    """读取纯文本 / Markdown"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


LOADERS = {
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".txt":  load_txt,
    ".md":   load_txt,
}


def load_document(file_path: str) -> str:
    """根据扩展名自动选择加载器"""
    ext = Path(file_path).suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"不支持的文件格式: {ext}，支持: {list(LOADERS.keys())}")
    return LOADERS[ext](file_path)


def load_directory(dir_path: str) -> list[dict]:
    """批量加载目录下的所有文档，返回 [{"name": ..., "content": ...}]"""
    docs = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in LOADERS:
                full = os.path.join(root, fname)
                docs.append({"name": fname, "content": load_document(full)})
    return docs
