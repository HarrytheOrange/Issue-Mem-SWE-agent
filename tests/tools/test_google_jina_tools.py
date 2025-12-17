import json
import os
from unittest import mock

from sweagent import TOOLS_DIR
from tests.utils import make_python_tool_importable


class _MockHeaders:
    def __init__(self, charset: str = "utf-8"):
        self._charset = charset

    def get_content_charset(self):  # noqa: ANN001 - mimic urllib interface
        return self._charset


class _MockResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = _MockHeaders()

    def read(self):  # noqa: ANN001 - mimic urllib interface
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001 - mimic urllib interface
        return False


def test_google_search_serper_main_prints_results(capsys):
    tool_path = TOOLS_DIR / "google_search" / "bin" / "google_search.py"
    make_python_tool_importable(tool_path, "google_search_bin")
    import google_search_bin  # type: ignore

    payload = {"organic": [{"title": "T1", "link": "https://example.com", "snippet": "S1"}]}
    body = json.dumps(payload).encode("utf-8")

    with mock.patch.dict(os.environ, {"SERPER_API_KEY": "dummy"}), mock.patch.object(
        google_search_bin.urllib.request, "urlopen", return_value=_MockResponse(body)
    ):
        rc = google_search_bin.main(["hello", "--topk", "1"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "[1] T1" in captured.out
    assert "https://example.com" in captured.out
    assert "S1" in captured.out


def test_google_search_missing_key_exits_2(capsys):
    tool_path = TOOLS_DIR / "google_search" / "bin" / "google_search.py"
    make_python_tool_importable(tool_path, "google_search_bin_2")
    import google_search_bin_2  # type: ignore

    with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(google_search_bin_2, "_find_dotenv", return_value=None):
        rc = google_search_bin_2.main(["hello"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Missing API key" in captured.err


def test_google_search_can_read_key_from_dotenv(capsys, tmp_path):
    tool_path = TOOLS_DIR / "google_search" / "bin" / "google_search.py"
    make_python_tool_importable(tool_path, "google_search_bin_dotenv")
    import google_search_bin_dotenv  # type: ignore

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("SERPER_API_KEY=dummy\n", encoding="utf-8")

    payload = {"organic": [{"title": "T1", "link": "https://example.com", "snippet": "S1"}]}
    body = json.dumps(payload).encode("utf-8")

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(google_search_bin_dotenv, "_find_dotenv", return_value=dotenv_path),
        mock.patch.object(google_search_bin_dotenv.urllib.request, "urlopen", return_value=_MockResponse(body)),
    ):
        rc = google_search_bin_dotenv.main(["hello", "--topk", "1", "--backend", "serper"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "[1] T1" in captured.out


def test_jina_read_builds_reader_url_and_truncates(capsys):
    tool_path = TOOLS_DIR / "jina_read" / "bin" / "jina_read.py"
    make_python_tool_importable(tool_path, "jina_read_bin")
    import jina_read_bin  # type: ignore

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001 - signature matches urllib
        assert req.get_full_url() == "https://r.jina.ai/https://example.com"
        assert req.get_header("Authorization") == "Bearer dummy"
        assert req.get_header("X-Api-Key") == "dummy"
        return _MockResponse(b"abcdefghijklmnopqrstuvwxyz")

    with mock.patch.dict(os.environ, {"JINA_API_KEY": "dummy"}), mock.patch.object(
        jina_read_bin.urllib.request, "urlopen", side_effect=_fake_urlopen
    ):
        rc = jina_read_bin.main(["https://example.com", "--max-chars", "10"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "abcdefg..."


def test_jina_read_can_read_key_from_dotenv(capsys, tmp_path):
    tool_path = TOOLS_DIR / "jina_read" / "bin" / "jina_read.py"
    make_python_tool_importable(tool_path, "jina_read_bin_dotenv")
    import jina_read_bin_dotenv  # type: ignore

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("JINA_API_KEY=dummy\n", encoding="utf-8")

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001 - signature matches urllib
        assert req.get_full_url() == "https://r.jina.ai/https://example.com"
        assert req.get_header("Authorization") == "Bearer dummy"
        assert req.get_header("X-Api-Key") == "dummy"
        return _MockResponse(b"abcdefghijklmnopqrstuvwxyz")

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(jina_read_bin_dotenv, "_find_dotenv", return_value=dotenv_path),
        mock.patch.object(jina_read_bin_dotenv.urllib.request, "urlopen", side_effect=_fake_urlopen),
    ):
        rc = jina_read_bin_dotenv.main(["https://example.com", "--max-chars", "10"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "abcdefg..."


