import os
import sys
import types
import importlib.util
import pathlib

import pytest
from fastapi.testclient import TestClient

# Ensure the server uses uvicorn to avoid gunicorn dependency during tests
os.environ.setdefault("YANDEX_SOLVER_USE_UVICORN", "1")

# Provide a lightweight stub for the main package to satisfy imports when loading
stub = types.ModuleType("yandex_captcha_puzzle_solver")
stub.Request = type("Request", (), {})
stub.Response = type("Response", (), {})
stub.Solver = type("Solver", (), {})
stub.proxy_controller = types.SimpleNamespace(ProxyController=type("ProxyController", (), {}))
sys.modules.setdefault("yandex_captcha_puzzle_solver", stub)

module_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "yandex_captcha_puzzle_solver" / "yandex_captcha_puzzle_solve_server.py"
spec = importlib.util.spec_from_file_location(
    "yandex_captcha_puzzle_solver.yandex_captcha_puzzle_solve_server", module_path
)
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)


@pytest.fixture
def client():
    return TestClient(server_module.server)


def test_get_token_success(monkeypatch, client):
    async def fake_process_solve_request(url, yandex_key, cookies=None, max_timeout=None, proxy=None):
        return server_module.HandleCommandResponse(
            status="ok",
            message="done",
            startTimestamp=1.0,
            endTimestamp=2.0,
            solution=server_module.HandleCommandResponseSolution(
                status="ok",
                url=url,
                cookies=[],
                user_agent="UA",
                token="token123",
            ),
        )

    monkeypatch.setattr(server_module, "process_solve_request", fake_process_solve_request)

    response = client.post(
        "/get_token",
        json={"url": "http://example.com", "yandex_key": "key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["solution"]["token"] == "token123"


def test_get_token_error(monkeypatch, client):
    async def fake_process_solve_request(*args, **kwargs):
        return server_module.HandleCommandResponse(
            status="error",
            message="Error: fail",
            startTimestamp=1.0,
            endTimestamp=2.0,
        )

    monkeypatch.setattr(server_module, "process_solve_request", fake_process_solve_request)

    response = client.post(
        "/get_token",
        json={"url": "http://example.com", "yandex_key": "key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "fail" in data["message"]


def test_get_token_missing_parameters(client):
    response = client.post(
        "/get_token",
        json={"url": "http://example.com"},  # missing yandex_key
    )
    assert response.status_code == 422
