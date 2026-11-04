"""#8816: the context-length lookup must not hand a bare built-in session a native custom twin's key.

With both ``providers.openai`` and ``providers['custom:openai']`` configured, the
lookup for a bare ``openai`` session selected the native custom record by slug.
The metadata lookup then sent the CUSTOM key to the BUILT-IN endpoint, and the
session borrowed the custom record's context limit. v0.52.544 used the built-in
record; exp-v0.52.597 (#8448) introduced the swap, and #8685's fix at
exp-v0.52.606 covered routing but not this lookup.

The rule (as in ``api.config._named_custom_provider_slug_for_provider``): a bare
built-in or plugin id is claimed only by a legacy ``custom_providers`` entry of
that name; a native ``custom:<id>`` record needs the explicit ``custom:<id>``.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from api import config, routes


def _config(pid, builtin_url="https://builtin.invalid/v1"):
    return {
        "model": {"provider": pid, "default": "stable-model"},
        "providers": {
            pid: {"base_url": builtin_url, "api_key": "synthetic-builtin"},
            "custom:" + pid: {
                "base_url": "https://named.invalid/v1",
                "api_key": "synthetic-named",
                "models": {"stable-model": {"context_length": 12345}},
            },
        },
    }


def _use(monkeypatch, data):
    monkeypatch.setattr(config, "cfg", data)
    monkeypatch.setattr(config, "get_config", lambda: data)


@pytest.mark.parametrize("pid", ["openai", "anthropic", "gemini"])
def test_a_bare_builtin_id_does_not_take_the_native_custom_record(monkeypatch, pid):
    data = _config(pid)
    _use(monkeypatch, data)

    builtin = routes._context_length_lookup_inputs_for_model("stable-model", pid, cfg=data)
    assert builtin.provider == pid
    assert builtin.base_url == "https://builtin.invalid/v1"
    assert builtin.api_key == "synthetic-builtin"
    assert builtin.config_context_length is None

    explicit = routes._context_length_lookup_inputs_for_model("stable-model", "custom:" + pid, cfg=data)
    assert explicit.base_url == "https://named.invalid/v1"
    assert explicit.api_key == "synthetic-named"
    assert explicit.config_context_length == 12345


def test_a_legacy_entry_named_after_a_builtin_still_claims_it(monkeypatch):
    """Unchanged from v0.52.544: a legacy ``custom_providers`` entry of that name keeps the bare id."""
    data = {
        "model": {"provider": "openai", "default": "stable-model"},
        "providers": {"openai": {"base_url": "https://builtin.invalid/v1", "api_key": "synthetic-builtin"}},
        "custom_providers": [{
            "name": "openai", "base_url": "https://legacy.invalid/v1", "api_key": "synthetic-legacy",
            "models": {"stable-model": {"context_length": 777}},
        }],
    }
    _use(monkeypatch, data)
    bare = routes._context_length_lookup_inputs_for_model("stable-model", "openai", cfg=data)
    assert (bare.api_key, bare.config_context_length) == ("synthetic-legacy", 777)


def test_the_builtin_endpoint_never_receives_the_custom_key(monkeypatch):
    """The real metadata lookup, against a loopback built-in endpoint that records Authorization."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            body = b'{"data":[{"id":"loopback-model","context_length":65536}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 (BaseHTTPRequestHandler's signature)
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_connect = socket.socket.connect

    def loopback_only(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            raise OSError("this test permits loopback only")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    try:
        builtin_url = f"http://127.0.0.1:{server.server_port}/v1"
        data = _config("openai", builtin_url)
        data["model"]["default"] = "loopback-model"
        _use(monkeypatch, data)
        routes._resolve_context_length_for_session_model("loopback-model", "openai", base_url=builtin_url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert seen, "the lookup made no request to the built-in endpoint"
    assert all(header == "Bearer synthetic-builtin" for header in seen), json.dumps(seen)
