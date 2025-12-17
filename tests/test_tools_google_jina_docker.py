from __future__ import annotations

import os
from unittest import mock

import pytest

from sweagent import TOOLS_DIR
from sweagent.tools.bundle import Bundle
from sweagent.tools.tools import ToolConfig, ToolHandler

from .conftest import swe_env_context


@pytest.mark.slow
def test_google_search_and_jina_read_available_in_docker(test_env_args):
    with mock.patch.dict(os.environ, {"SERPER_API_KEY": "dummy", "JINA_API_KEY": "dummy"}):
        tools = ToolHandler.from_config(
            ToolConfig(
                enable_bash_tool=False,
                bundles=[
                    Bundle(path=TOOLS_DIR / "google_search"),
                    Bundle(path=TOOLS_DIR / "jina_read"),
                ],
            )
        )

        with swe_env_context(test_env_args) as env:
            tools.install(env)
            out = env.communicate("google_search --help", check="raise")
            assert "Google search tool" in out
            out = env.communicate("jina_read --help", check="raise")
            assert "Jina Reader" in out

            out = env.communicate(
                "python3 - <<'PY'\n"
                "import os\n"
                "print('SERPER_OK' if os.environ.get('SERPER_API_KEY') == 'dummy' else 'SERPER_MISSING')\n"
                "print('JINA_OK' if os.environ.get('JINA_API_KEY') == 'dummy' else 'JINA_MISSING')\n"
                "PY",
                check="raise",
            )
            assert "SERPER_OK" in out
            assert "JINA_OK" in out


