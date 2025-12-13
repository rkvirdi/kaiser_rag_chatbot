# tests/test_data_loader.py

from pathlib import Path

def test_load_raw_documents(monkeypatch, tmp_path):
    """
    Test that load_raw_documents:
    - iterates over PDF paths returned by list_pdf_files
    - uses PyPDFLoader to load pages
    - returns dicts with text + metadata including source/page/doc_type
    """
    import backend.data_ingestion.data_loading as data_loader

    # --- 1) Monkeypatch list_pdf_files to avoid real filesystem ---
    fake_pdf_paths = [
        tmp_path / "compliance" / "doc1.pdf",
        tmp_path / "policy" / "plan1.pdf",
    ]
    (tmp_path / "compliance").mkdir()
    (tmp_path / "policy").mkdir()
    for p in fake_pdf_paths:
        p.touch()  # create empty files so Path exists

    def fake_list_pdf_files(_):
        return fake_pdf_paths

    monkeypatch.setattr(data_loader, "list_pdf_files", fake_list_pdf_files)

    # --- 2) Monkeypatch PyPDFLoader to avoid real PDF parsing ---
    class FakePage:
        def __init__(self, text, metadata):
            self.page_content = text
            self.metadata = metadata

    class FakeLoader:
        def __init__(self, path_str):
            self.path_str = path_str

        def load(self):
            # Return 2 fake pages per PDF
            return [
                FakePage(
                    text=f"Page 0 content for {self.path_str}",
                    metadata={"page": 0},
                ),
                FakePage(
                    text=f"Page 1 content for {self.path_str}",
                    metadata={"page": 1},
                ),
            ]

    # Replace PyPDFLoader in data_loader module
    monkeypatch.setattr(data_loader, "PyPDFLoader", FakeLoader)
    monkeypatch.setattr(data_loader, "_HAS_LANGCHAIN", True)

    # --- 3) Call load_raw_documents ---
    docs = data_loader.load_raw_documents(tmp_path)  # raw_dir arg is ignored in fake_list_pdf_files

    # --- 4) Assertions ---
    # 2 PDFs * 2 pages each = 4 docs
    assert len(docs) == 4

    # Check structure of first doc
    d0 = docs[0]
    assert "text" in d0
    assert "metadata" in d0
    assert isinstance(d0["text"], str)
    meta = d0["metadata"]
    assert "source" in meta
    assert "page" in meta
    assert "doc_type" in meta

    # Check that doc_type inference works
    # One path had "compliance" in it, one had "policy"
    doc_types = {d["metadata"]["doc_type"] for d in docs}
    assert "compliance" in doc_types
    assert "benefit_policy" in doc_types
