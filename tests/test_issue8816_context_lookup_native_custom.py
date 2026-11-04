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


def _xai_config(active):
    return {
        "model": {"provider": active, "default": "grok-x"},
        "providers": {
            "x-ai": {"base_url": "https://builtin.invalid/v1", "api_key": "synthetic-builtin"},
            "custom:xai": {
                "base_url": "https://named.invalid/v1",
                "api_key": "synthetic-named",
                "models": {"grok-x": {"context_length": 12345}},
            },
        },
    }


@pytest.mark.parametrize("active", ["x-ai", "xai", "grok"])
def test_an_xai_alias_stays_builtin_without_plugin_discovery(monkeypatch, active):
    """With the plugin registry empty, the Agent's canonical ``xai`` is not in WebUI's
    static tables (they say ``x-ai``); it must still count as a built-in, for the
    context lookup and for routing, as on v0.52.544."""
    from api import plugin_providers

    monkeypatch.setattr(plugin_providers, "_PROFILES_BY_NAME", {})
    data = _xai_config(active)
    _use(monkeypatch, data)

    bare = routes._context_length_lookup_inputs_for_model("grok-x", active, cfg=data)
    assert (bare.api_key, bare.config_context_length) == ("synthetic-builtin", None)
    assert config._named_custom_provider_slug_for_provider(active, data) == ""
    _model, provider, base_url = config.resolve_model_provider("grok-x")
    assert not str(provider).startswith("custom:"), (provider, base_url)
    assert base_url != "https://named.invalid/v1"

    explicit = routes._context_length_lookup_inputs_for_model("grok-x", "custom:xai", cfg=data)
    assert (explicit.api_key, explicit.config_context_length) == ("synthetic-named", 12345)


def _family_config(active, record, builtin="fireworks"):
    return {
        "model": {"provider": active, "default": "m-x"},
        "providers": {
            builtin: {"base_url": "https://builtin.invalid/v1", "api_key": "synthetic-builtin"},
            "custom:" + record: {
                "base_url": "https://named.invalid/v1",
                "api_key": "synthetic-named",
                "models": {"m-x": {"context_length": 12345}},
            },
        },
    }


@pytest.mark.parametrize("discovery", ["as-is", "empty"])
@pytest.mark.parametrize(
    "active,record",
    [
        ("fireworks", "fireworks"),
        ("Fireworks", "fireworks"),
        ("fw", "fw"),
        ("fw", "fireworks"),
        ("fireworks-ai", "fireworks-ai"),
        ("fireworks_ai", "fireworks_ai"),
    ],
)
def test_an_agent_catalog_builtin_stays_builtin_in_both_discovery_states(monkeypatch, discovery, active, record):
    """Fireworks is outside WebUI's static tables; the Agent's provider catalog and
    alias table still make every bare spelling a built-in, as on v0.52.544."""
    from api import plugin_providers

    if discovery == "empty":
        monkeypatch.setattr(plugin_providers, "_PROFILES_BY_NAME", {})
    data = _family_config(active, record)
    _use(monkeypatch, data)

    assert config._is_bare_builtin_provider_id(active)
    bare = routes._context_length_lookup_inputs_for_model("m-x", active, cfg=data)
    assert bare.api_key != "synthetic-named"
    assert bare.config_context_length != 12345
    assert config._named_custom_provider_slug_for_provider(active, data) == ""
    _model, provider, base_url = config.resolve_model_provider("m-x")
    assert not str(provider).startswith("custom:"), (provider, base_url)
    assert base_url != "https://named.invalid/v1"

    explicit = routes._context_length_lookup_inputs_for_model("m-x", "custom:" + record, cfg=data)
    assert (explicit.api_key, explicit.config_context_length) == ("synthetic-named", 12345)


@pytest.mark.parametrize("discovery", ["as-is", "empty"])
def test_a_genuine_bare_custom_id_still_takes_its_native_record(monkeypatch, discovery):
    """Control: a name that is no built-in keeps #8689's behaviour on .630."""
    from api import plugin_providers

    if discovery == "empty":
        monkeypatch.setattr(plugin_providers, "_PROFILES_BY_NAME", {})
    data = {
        "model": {"provider": "my-proxy", "default": "m-x"},
        "providers": {
            "custom:my-proxy": {
                "base_url": "https://named.invalid/v1",
                "api_key": "synthetic-named",
                "models": {"m-x": {"context_length": 12345}},
            },
        },
    }
    _use(monkeypatch, data)

    assert not config._is_bare_builtin_provider_id("my-proxy")
    found = routes._context_length_lookup_inputs_for_model("m-x", "my-proxy", cfg=data)
    assert (found.api_key, found.config_context_length) == ("synthetic-named", 12345)
    assert config._named_custom_provider_slug_for_provider("my-proxy", data) == "custom:my-proxy"


def test_an_agent_without_the_provider_catalog_is_tolerated(monkeypatch):
    """An Agent whose ``hermes_cli.models`` lacks ``CANONICAL_PROVIDERS`` gives no catalog
    ids rather than an error; WebUI's own tables still classify the built-ins they know."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "hermes_cli.models", types.ModuleType("hermes_cli.models"))
    assert config._agent_builtin_provider_ids() == ()
    assert config._is_bare_builtin_provider_id("openai")
    assert not config._is_bare_builtin_provider_id("my-proxy")
    assert not config._is_bare_builtin_provider_id("custom:openai")


class _RaisingSlug:
    @property
    def slug(self):
        raise RuntimeError("catalog entry from a future Agent")


@pytest.mark.parametrize(
    "catalog,expected",
    [
        (5, ()),
        ([_RaisingSlug()], ()),
        (["Fireworks", "", None, ("xai", "xAI", "")], ("fireworks", "xai")),
        ([{"slug": "fireworks"}], ()),
    ],
)
def test_an_unexpected_catalog_shape_never_raises_into_routing(monkeypatch, catalog, expected):
    import sys
    import types

    module = types.ModuleType("hermes_cli.models")
    module.CANONICAL_PROVIDERS = catalog
    monkeypatch.setitem(sys.modules, "hermes_cli.models", module)
    assert config._agent_builtin_provider_ids() == expected
    assert config._is_bare_builtin_provider_id("openai")
    assert not config._is_bare_builtin_provider_id("my-proxy")


@pytest.mark.parametrize("discovery", ["as-is", "empty"])
@pytest.mark.parametrize("active", ["vllm", "local"])
@pytest.mark.parametrize("shape", ["bare-native-key", "custom-key"])
def test_a_generic_custom_alias_keeps_its_own_record(monkeypatch, discovery, active, shape):
    """``vllm``/``local`` alias the generic ``custom`` placeholder in the Agent's table, not
    a built-in: with its own record, the bare id keeps .630's route to that record.
    (``ollama`` is ``custom`` in WebUI's own alias table too, so .630 never routed it to a
    native record; the fold leaves that unchanged.)"""
    from api import plugin_providers

    if discovery == "empty":
        monkeypatch.setattr(plugin_providers, "_PROFILES_BY_NAME", {})
    key = active if shape == "bare-native-key" else "custom:" + active
    data = {
        "model": {"provider": active, "default": "m-x"},
        "providers": {key: {"base_url": "https://named.invalid/v1", "api_key": "synthetic-named", "models": {"m-x": {}}}},
    }
    _use(monkeypatch, data)

    assert not config._is_bare_builtin_provider_id(active)
    assert config._named_custom_provider_slug_for_provider(active, data) == "custom:" + active
    _model, provider, base_url = config.resolve_model_provider("m-x")
    assert (provider, base_url) == ("custom:" + active, "https://named.invalid/v1")


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
