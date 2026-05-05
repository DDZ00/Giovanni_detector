"""Tests for /samples/batch — multi-file upload endpoint.

Storage is monkeypatched so tests don't touch a real SQLite file. We only
need to prove the route correctly:
  - rejects non-multipart requests
  - iterates files and aggregates per-file outcomes
  - keeps going when one file fails (parse error / duplicate)
"""
import io

import pytest


@pytest.fixture
def client(monkeypatch):
    from app import app as flask_app, storage as app_storage

    seq = {"id": 0}

    def fake_add_sample(text, *, label=None, english_proficiency=None,
                        source=None, filename=None, notes=None, db_path=None):
        if not text.strip():
            raise ValueError("text is empty")
        if filename and "DUP" in (filename or ""):
            raise app_storage.DuplicateSample("sample already exists with id=42")
        seq["id"] += 1
        return {
            "id": seq["id"], "label": label,
            "english_proficiency": english_proficiency,
            "source": source, "filename": filename,
            "n_words": len(text.split()), "sha256": "deadbeef", "notes": notes,
            "created_at": "2026-05-04T00:00:00+00:00",
        }

    monkeypatch.setattr("app.storage.add_sample", fake_add_sample)
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _txt(name: str, body: str = "hello world from giovanni"):
    return (io.BytesIO(body.encode("utf-8")), name)


class TestBatchUpload:
    def test_rejects_non_multipart(self, client):
        r = client.post("/samples/batch", json={"files": []})
        assert r.status_code == 400
        assert "multipart" in r.get_json()["error"].lower()

    def test_rejects_when_no_files(self, client):
        r = client.post("/samples/batch", data={}, content_type="multipart/form-data")
        assert r.status_code == 400
        assert "no files" in r.get_json()["error"]

    def test_adds_multiple_txt_files(self, client):
        data = {"files": [_txt("a.txt"), _txt("b.txt", "another body of text")]}
        r = client.post("/samples/batch", data=data, content_type="multipart/form-data")
        assert r.status_code == 201
        body = r.get_json()
        assert body["added"] == 2
        assert body["total"] == 2
        assert [e["status"] for e in body["results"]] == ["added", "added"]
        assert body["results"][0]["filename"] == "a.txt"

    def test_continues_past_unsupported_extension(self, client):
        # The middle file has an unknown extension — extract_text raises ValueError.
        # Batch must still process the surrounding files.
        data = {"files": [_txt("good.txt"), _txt("bad.xyz"), _txt("also_good.txt")]}
        r = client.post("/samples/batch", data=data, content_type="multipart/form-data")
        body = r.get_json()
        assert body["added"] == 2
        assert body["total"] == 3
        statuses = [e["status"] for e in body["results"]]
        assert statuses == ["added", "error", "added"]
        assert "unsupported file extension" in body["results"][1]["error"]

    def test_marks_duplicates_without_aborting(self, client):
        data = {"files": [_txt("DUP_already_seen.txt"), _txt("fresh.txt")]}
        r = client.post("/samples/batch", data=data, content_type="multipart/form-data")
        body = r.get_json()
        assert body["added"] == 1
        assert body["results"][0]["status"] == "duplicate"
        assert body["results"][1]["status"] == "added"

    def test_label_and_source_apply_to_all(self, client):
        captured = []
        from app import storage as app_storage
        original = app_storage.add_sample

        def spy(text, *, label=None, english_proficiency=None,
                source=None, filename=None, notes=None, db_path=None):
            captured.append({"label": label, "english_proficiency": english_proficiency,
                             "source": source, "filename": filename})
            return {"id": len(captured), "label": label,
                    "english_proficiency": english_proficiency,
                    "source": source, "filename": filename, "n_words": 1,
                    "sha256": "x", "notes": notes, "created_at": "t"}
        app_storage.add_sample = spy
        try:
            data = {
                "files": [_txt("x.txt"), _txt("y.txt")],
                "label": "human", "source": "Corpus-A",
                "english_proficiency": "non_native",
            }
            r = client.post("/samples/batch", data=data, content_type="multipart/form-data")
            assert r.status_code == 201
            assert all(c["label"] == "human" for c in captured)
            assert all(c["source"] == "Corpus-A" for c in captured)
            assert all(c["english_proficiency"] == "non_native" for c in captured)
        finally:
            app_storage.add_sample = original

    def test_source_defaults_to_filename_when_blank(self, client):
        captured = []
        from app import storage as app_storage
        original = app_storage.add_sample

        def spy(text, *, label=None, english_proficiency=None,
                source=None, filename=None, notes=None, db_path=None):
            captured.append({"source": source, "filename": filename})
            return {"id": len(captured), "label": label,
                    "english_proficiency": english_proficiency,
                    "source": source, "filename": filename, "n_words": 1,
                    "sha256": "x", "notes": notes, "created_at": "t"}
        app_storage.add_sample = spy
        try:
            data = {"files": [_txt("paper-2024.txt")]}
            client.post("/samples/batch", data=data, content_type="multipart/form-data")
            assert captured[0]["source"] == "paper-2024.txt"
        finally:
            app_storage.add_sample = original

    def test_english_proficiency_omitted_when_blank(self, client):
        captured = []
        from app import storage as app_storage
        original = app_storage.add_sample

        def spy(text, *, label=None, english_proficiency=None,
                source=None, filename=None, notes=None, db_path=None):
            captured.append({"english_proficiency": english_proficiency})
            return {"id": len(captured), "label": label,
                    "english_proficiency": english_proficiency,
                    "source": source, "filename": filename, "n_words": 1,
                    "sha256": "x", "notes": notes, "created_at": "t"}
        app_storage.add_sample = spy
        try:
            # No english_proficiency in form → must reach storage as None,
            # not as the empty string "".
            data = {"files": [_txt("solo.txt")]}
            client.post("/samples/batch", data=data, content_type="multipart/form-data")
            assert captured[0]["english_proficiency"] is None
        finally:
            app_storage.add_sample = original


class TestPdfExtraction:
    """Pure-function tests for the new PDF branch in extract_text."""

    def test_unsupported_extension_raises(self):
        from modules.docx_parse import extract_text
        with pytest.raises(ValueError, match="unsupported file extension"):
            extract_text("oops.xyz", b"\x00\x01")

    def test_pdf_with_no_text_raises_clearly(self):
        # A minimal "PDF" with no extractable text — pypdf returns empty pages
        # and our guard converts that into a clear ValueError instead of a
        # silent empty sample.
        from modules.docx_parse import extract_text
        # Smallest valid empty PDF (one blank page)
        empty_pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000052 00000 n\n0000000101 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF\n"
        )
        with pytest.raises(ValueError, match="no extractable text"):
            extract_text("blank.pdf", empty_pdf)
