"""Tests de l'API HTTP (FastAPI + TestClient, backend scripté, bases de savoir
réelles sous tmp_path avec l'embedder hash, sans réseau)."""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from dhaos import __version__
from dhaos.agent.session import SessionStore
from dhaos.backends.base import Backend, BackendError
from dhaos.config import Settings
from dhaos.kb.manager import KnowledgeManager
from dhaos.policy import Journal
from dhaos.types import ChatResponse

from .fakes import FakeBackend, tool_call, tool_response

# --------------------------------------------------------------- outillage


class BackendFactory:
    """Fabrique enregistrant ses appels ; les réponses scriptées sont
    partagées entre les backends créés (consommées dans l'ordre)."""

    def __init__(self, responses: list[str | ChatResponse] | None = None) -> None:
        self.responses: list[str | ChatResponse] = list(responses or [])
        self.calls: list[tuple[str | None, str | None]] = []
        self.backends: list[FakeBackend] = []

    def __call__(self, settings: Settings, name: str | None, model: str | None) -> Backend:
        self.calls.append((name, model))
        backend = FakeBackend()
        backend.responses = self.responses  # liste partagée
        self.backends.append(backend)
        return backend


def parse_sse(text: str) -> list[tuple[str, Any]]:
    """Découpe un flux ``event:``/``data:`` en couples (type, JSON)."""
    events: list[tuple[str, Any]] = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines or all(ln.startswith(":") for ln in lines):
            continue
        event = ""
        data_lines: list[str] = []
        for line in lines:
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        events.append((event, json.loads("\n".join(data_lines)) if data_lines else None))
    return events


LOCAL_BASE_URL = "http://127.0.0.1"


def local_client(app: Any, *, raise_server_exceptions: bool = True) -> TestClient:
    """Client de test local (``Host: 127.0.0.1``) portant le jeton de l'application
    (``api.token`` ou jeton généré) — le comportement par défaut d'un client
    légitime ; les tests d'authentification retirent l'en-tête explicitement."""
    client = TestClient(app, base_url=LOCAL_BASE_URL, raise_server_exceptions=raise_server_exceptions)
    client.headers["Authorization"] = f"Bearer {app.state.token}"
    return client


def make_client(
    settings: Settings,
    responses: list[str | ChatResponse] | None = None,
    *,
    backend_factory: Callable[..., Backend] | None = None,
    raise_server_exceptions: bool = True,
) -> tuple[TestClient, BackendFactory | None]:
    from dhaos.api.server import create_app

    factory = None if backend_factory is not None else BackendFactory(responses)
    app = create_app(settings, backend_factory=backend_factory or factory, kb_factory=KnowledgeManager)
    return local_client(app, raise_server_exceptions=raise_server_exceptions), factory


def without_auth(client: TestClient) -> dict[str, str]:
    """En-têtes annulant le jeton porté par défaut par ``local_client``."""
    return {"Authorization": ""}


@pytest.fixture
def api(settings: Settings):
    """Client + fabrique, bases ouvertes/fermées par le cycle de vie."""
    client, factory = make_client(settings, ["Bonjour !"])
    with client:
        yield client, factory


# ------------------------------------------------------------------ health


def test_health_reports_backend_and_kb(api) -> None:
    client, _ = api
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["backend"]["name"] == "fake"
    assert body["backend"]["model"] == "fake-model"
    assert body["backend"]["health"]["ok"] is True
    assert body["kb"] == {"bases": 0, "documents": 0, "chunks": 0}


def test_health_survives_backend_failure(settings: Settings) -> None:
    def broken(settings: Settings, name: str | None, model: str | None) -> Backend:
        raise BackendError("Ollama injoignable")

    client, _ = make_client(settings, backend_factory=broken)
    with client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"]["name"] == settings.backends.default
    assert body["backend"]["health"]["ok"] is False
    assert "injoignable" in body["backend"]["health"]["detail"]


def test_health_without_token_is_minimal(settings: Settings) -> None:
    """Régression : ``/health`` sans jeton ne construit aucun backend (aucun
    appel sortant) et ne révèle ni hôte, ni modèles, ni message d'erreur."""
    settings.api.token = "secret"
    client, factory = make_client(settings)
    with client:
        client.post("/kb", json={"name": "dev"})
        r = client.get("/health", headers=without_auth(client))
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["version"] == __version__
        assert body["backend"] == {"name": settings.backends.default, "model": "", "health": {}}
        assert body["kb"] == {"bases": 0, "documents": 0, "chunks": 0}
        assert factory.calls == []  # aucun backend construit
        r = client.get("/health", headers={"Authorization": "Bearer mauvais"})
        assert r.status_code == 200 and r.json()["backend"]["health"] == {} and factory.calls == []
        # Avec le jeton : détail complet.
        r = client.get("/health", headers={"Authorization": "Bearer secret"})
        body = r.json()
        assert body["backend"]["health"]["ok"] is True and body["backend"]["model"] == "fake-model"
        assert body["kb"]["bases"] == 1
        assert factory.calls == [(None, None)]


def test_health_generated_token_gives_full_detail(api) -> None:
    """Sans ``api.token`` configuré, le jeton généré donne le détail complet
    et son absence l'état minimal."""
    client, factory = api
    assert client.get("/health").json()["backend"]["health"]["ok"] is True
    assert client.get("/health", headers=without_auth(client)).json()["backend"]["health"] == {}
    assert factory.calls == [(None, None)]


def test_kb_closed_at_shutdown(settings: Settings) -> None:
    client, _ = make_client(settings)
    app = client.app
    with client:
        assert isinstance(app.state.kb, KnowledgeManager)
    assert app.state.kb is None


# -------------------------------------------------------------------- auth


def test_auth_required_when_token_set(settings: Settings) -> None:
    settings.api.token = "s3cret"
    client, _ = make_client(settings)
    assert client.app.state.token == "s3cret" and client.app.state.token_generated is False
    with client:
        bare = without_auth(client)
        assert client.get("/health", headers=bare).status_code == 200
        r = client.get("/kb", headers=bare)
        assert r.status_code == 401
        assert "detail" in r.json()
        assert r.headers.get("www-authenticate") == "Bearer"
        assert client.get("/kb", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/kb", headers={"Authorization": "Basic s3cret"}).status_code == 401
        assert client.get("/config", headers={"Authorization": "bearer s3cret"}).status_code == 200
        r = client.get("/kb", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200 and r.json() == []
        r = client.post("/chat", json={"message": "x", "stream": False}, headers=bare)
        assert r.status_code == 401


def test_auth_non_ascii_header_is_401(settings: Settings) -> None:
    """Régression : un octet non ASCII dans l'en-tête ne doit pas produire un
    500 (``compare_digest`` refuse les ``str`` non ASCII) mais un 401."""
    settings.api.token = "s3cret"
    client, _ = make_client(settings, raise_server_exceptions=False)
    with client:
        for header in (b"Bearer \xe9s3cret", b"Bearer s3cret\xc3\xa9", "Bearer s3crét".encode("latin-1")):
            r = client.get("/kb", headers={"Authorization": header})
            assert r.status_code == 401, header
            assert r.headers.get("www-authenticate") == "Bearer"
            assert "erreur interne" not in r.text
        assert client.get("/kb", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_api_requires_token_by_default(settings: Settings) -> None:
    """Régression : sans ``api.token``, un jeton aléatoire est généré ; rien
    ne passe sans lui (sauf l'état minimal de ``/health``)."""
    assert settings.api.token is None
    client, _ = make_client(settings)
    app = client.app
    token = app.state.token
    assert app.state.token_generated is True
    assert isinstance(token, str) and len(token) >= 32
    assert settings.api.token is None  # la configuration n'est pas modifiée
    with client:
        bare = without_auth(client)
        assert client.get("/config", headers=bare).status_code == 401
        assert client.get("/kb", headers=bare).status_code == 401
        assert client.get("/sessions", headers=bare).status_code == 401
        r = client.post("/chat", json={"message": "x", "stream": False}, headers=bare)
        assert r.status_code == 401 and r.headers.get("www-authenticate") == "Bearer"
        assert client.get("/health", headers=bare).status_code == 200
        assert client.get("/config", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        assert client.get("/config").json()["api"]["token"] is None  # jamais renvoyé par /config
        assert token not in client.get("/config").text
    # Deux applications ne partagent pas le même jeton.
    other, _ = make_client(settings)
    assert other.app.state.token != token


def test_api_rejects_foreign_host(settings: Settings) -> None:
    """Régression : un en-tête ``Host`` inattendu (DNS rebinding) ⇒ 400, même
    avec le jeton et même sur ``/health``."""
    settings.api.token = "s3cret"
    client, _ = make_client(settings)
    with client:
        for host in ("evil.example.com", "evil.example.com:8642", ""):
            r = client.get("/config", headers={"Host": host})
            assert r.status_code == 400, host
            assert "Host" in r.json()["detail"]
            assert client.get("/health", headers={"Host": host}).status_code == 400
        for host in ("127.0.0.1", "127.0.0.1:8642", "localhost", "LOCALHOST:1", "[::1]:8642", "[::1]"):
            assert client.get("/config", headers={"Host": host}).status_code == 200, host
        assert client.get("/health", headers={"Host": "localhost:8642"}).status_code == 200


def test_api_allowed_hosts_setting(settings: Settings) -> None:
    settings.api.host = "0.0.0.0"
    settings.api.allowed_hosts = ["api.exemple.local"]
    client, _ = make_client(settings)
    with client:
        assert client.get("/kb", headers={"Host": "api.exemple.local:9000"}).status_code == 200
        assert client.get("/kb", headers={"Host": "127.0.0.1"}).status_code == 400
    settings.api.allowed_hosts = ["*"]
    client, _ = make_client(settings)
    with client:
        assert client.get("/kb", headers={"Host": "n-importe-quoi.example"}).status_code == 200


def test_api_rejects_foreign_origin(settings: Settings) -> None:
    """Régression : un en-tête ``Origin`` non listé ⇒ 403 (aucun client
    navigateur par défaut) ; sans ``Origin`` la requête passe."""
    settings.api.token = "s3cret"
    client, _ = make_client(settings)
    with client:
        r = client.post("/chat", json={"message": "x", "stream": False}, headers={"Origin": "http://evil.example.com"})
        assert r.status_code == 403
        assert "origine" in r.json()["detail"]
        assert client.get("/kb", headers={"Origin": "http://127.0.0.1:8642"}).status_code == 403
        assert client.get("/kb", headers={"Origin": "null"}).status_code == 403
        assert client.get("/kb").status_code == 200
        assert client.get("/sessions").json() == []  # aucune session créée par le /chat refusé
    settings.api.allowed_origins = ["http://localhost:3000/"]
    client, _ = make_client(settings)
    with client:
        assert client.get("/kb", headers={"Origin": "http://LOCALHOST:3000"}).status_code == 200
        assert client.get("/kb", headers={"Origin": "http://localhost:3001"}).status_code == 403


def test_chat_needs_token_to_write_project(settings: Settings, project_root: Path) -> None:
    """Régression : un ``/chat`` piloté depuis un hôte tiers et sans jeton ne
    doit ni écrire dans le projet, ni exécuter de commande."""
    target = project_root / "pwned.txt"
    responses: list[str | ChatResponse] = [
        tool_response(tool_call("write_file", "c1", path=str(target), content="écrit via l'API")),
        tool_response(tool_call("run_command", "c2", command="echo commande-executee")),
        "fini",
    ]
    client, factory = make_client(settings, responses)
    with client:
        evil = {"Host": "evil.example.com", "Origin": "http://evil.example.com", "Authorization": ""}
        assert client.post("/chat", json={"message": "x", "stream": False}, headers=evil).status_code == 400
        evil.pop("Host")
        assert client.post("/chat", json={"message": "x", "stream": False}, headers=evil).status_code == 403
        evil.pop("Origin")
        assert client.post("/chat", json={"message": "x", "stream": False}, headers=evil).status_code == 401
        assert not target.exists()
        assert factory.calls == [] and client.get("/sessions").json() == []
        # Le client légitime (jeton généré, hôte local) peut, lui, écrire dans le projet.
        r = client.post("/chat", json={"message": "x", "stream": False})
        assert r.status_code == 200 and r.json()["tool_calls"] == 2
        assert target.read_text(encoding="utf-8") == "écrit via l'API"


def test_host_guard_helpers() -> None:
    from dhaos.api.server import allowed_hosts_for, host_name, normalize_origin

    assert host_name("Example.COM:8642") == "example.com"
    assert host_name("[::1]:8642") == "::1" and host_name("[::1]") == "::1"
    assert host_name("127.0.0.1") == "127.0.0.1" and host_name("") == ""
    assert normalize_origin("HTTP://Localhost:3000/") == "http://localhost:3000"
    s = Settings()
    assert allowed_hosts_for(s) == {"localhost", "127.0.0.1", "::1"}
    s.api.host = "192.168.1.10"
    assert "192.168.1.10" in allowed_hosts_for(s)
    s.api.allowed_hosts = ["Proxy.Local:443"]
    assert allowed_hosts_for(s) == {"proxy.local"}


# ------------------------------------------------------------------ config


def test_config_masks_secrets(settings: Settings) -> None:
    settings.api.token = "s3cret"
    settings.web.brave_api_key = "brave-key"
    client, _ = make_client(settings)
    with client:
        r = client.get("/config", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200
    body = r.json()
    assert body["api"]["token"] == "***"
    assert body["web"]["brave_api_key"] == "***"
    assert body["kb"]["embedder"] == "hash"
    assert body["backends"]["default"] == settings.backends.default
    assert isinstance(body["source_path"], str) and isinstance(body["project_root"], str)  # utilisés par l'interface
    assert "s3cret" not in r.text and "brave-key" not in r.text


def test_config_unset_secret_stays_null(api) -> None:
    client, _ = api
    r = client.get("/config")
    assert r.json()["api"]["token"] is None
    assert client.app.state.token not in r.text


# ---------------------------------------------------------------------- kb


def test_kb_crud(api) -> None:
    client, _ = api
    r = client.post("/kb", json={"name": "Dev Base", "description": "base développeur"})
    assert r.status_code == 201, r.text
    info = r.json()
    assert info["name"] == "dev-base"
    assert info["description"] == "base développeur"
    assert info["n_docs"] == 0 and info["embedder"].startswith("hash:")

    assert client.post("/kb", json={"name": "dev-base"}).status_code == 400
    assert client.post("/kb", json={"name": "bad name!"}).status_code == 400
    assert client.post("/kb", json={"name": ""}).status_code == 422
    assert client.post("/kb", json={}).status_code == 422

    r = client.get("/kb")
    assert r.status_code == 200
    assert [b["name"] for b in r.json()] == ["dev-base"]

    r = client.get("/kb/dev-base")
    assert r.status_code == 200
    assert r.json()["documents"] == []
    assert client.get("/kb/absente").status_code == 404
    assert client.get("/kb/bad%20name!").status_code == 400

    r = client.patch("/kb/dev-base", json={"description": "nouvelle description"})
    assert r.status_code == 200 and r.json()["description"] == "nouvelle description"
    r = client.patch("/kb/dev-base", json={"new_name": "developpeur"})
    assert r.status_code == 200 and r.json()["name"] == "developpeur"
    assert r.json()["description"] == "nouvelle description"
    assert client.get("/kb/dev-base").status_code == 404
    assert client.patch("/kb/absente", json={"description": "x"}).status_code == 404

    client.post("/kb", json={"name": "autre"})
    assert client.patch("/kb/developpeur", json={"new_name": "autre"}).status_code == 400

    assert client.delete("/kb/developpeur").status_code == 204
    assert client.delete("/kb/developpeur").status_code == 404
    assert [b["name"] for b in client.get("/kb").json()] == ["autre"]


def test_kb_notes_and_search(api) -> None:
    client, _ = api
    client.post("/kb", json={"name": "dev"})
    r = client.post("/kb/dev/notes", json={"text": "pytest : les fixtures isolent l'état des tests.", "title": "Pytest"})
    assert r.status_code == 200, r.text
    doc_id = r.json()["doc_id"]
    assert isinstance(doc_id, int) and doc_id > 0
    r = client.post("/kb/dev/notes", json={"text": "Docker compose orchestre plusieurs conteneurs."})
    assert r.status_code == 200
    assert client.post("/kb/absente/notes", json={"text": "x"}).status_code == 404
    assert client.post("/kb/dev/notes", json={"text": "   "}).status_code == 400
    assert client.post("/kb/dev/notes", json={}).status_code == 422

    detail = client.get("/kb/dev").json()
    assert detail["n_docs"] == 2
    assert len(detail["documents"]) == 2
    by_title = {d["title"]: d for d in detail["documents"]}
    assert by_title["Pytest"]["source"] == "note:Pytest"
    assert by_title["Pytest"]["id"] == doc_id

    r = client.post("/kb/search", json={"query": "fixtures pytest"})
    assert r.status_code == 200, r.text
    hits = r.json()
    assert hits and hits[0]["base"] == "dev"
    assert hits[0]["doc_id"] == doc_id
    assert "fixtures" in hits[0]["text"]
    assert set(hits[0]) >= {"base", "source", "title", "text", "score", "chunk_ord", "doc_id"}

    r = client.post("/kb/search", json={"query": "conteneurs", "bases": ["dev"], "top_k": 1, "mode": "keyword"})
    assert r.status_code == 200 and len(r.json()) == 1
    assert "Docker" in r.json()[0]["text"]

    assert client.post("/kb/search", json={"query": "x", "bases": ["absente"]}).status_code == 404
    assert client.post("/kb/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/kb/search", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/kb/search", json={"query": ""}).status_code == 422


def test_kb_documents_ingest_and_remove(api, settings: Settings, project_root: Path) -> None:
    client, _ = api
    client.post("/kb", json={"name": "docs"})
    readme = project_root / "README.md"
    readme.write_text("# Projet\n\nInstallation avec pip install -e . puis pytest.\n", encoding="utf-8")
    (project_root / "notes.txt").write_text("Le déploiement passe par systemd.\n", encoding="utf-8")

    r = client.post("/kb/docs/documents", json={"sources": [str(project_root)], "recursive": True})
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["base"] == "docs"
    assert report["added"] == 2 and report["failed"] == 0
    assert report["chunks"] >= 2
    assert "ajouté" in report["summary"]

    # Ré-ingestion : inchangé ⇒ ignoré.
    report = client.post("/kb/docs/documents", json={"sources": [str(readme)]}).json()
    assert report["added"] == 0 and report["skipped"] == 1

    # Source introuvable ⇒ échec compté, pas d'erreur HTTP.
    report = client.post("/kb/docs/documents", json={"sources": [str(project_root / "absent.md")]}).json()
    assert report["failed"] == 1 and report["errors"]

    # Chemin protégé par la politique ⇒ refusé sans lecture.
    secret = project_root / "id_rsa"
    secret.write_text("PRIVATE", encoding="utf-8")
    report = client.post("/kb/docs/documents", json={"sources": [str(secret)]}).json()
    assert report["failed"] == 1 and "protégé" in report["errors"][0]
    assert all(d["source"] != str(secret) for d in client.get("/kb/docs").json()["documents"])

    assert client.post("/kb/absente/documents", json={"sources": ["x"]}).status_code == 404
    assert client.post("/kb/docs/documents", json={"sources": []}).status_code == 422
    # Un dossier dont le nom est protégé n'est pas parcouru.
    (project_root / ".env").mkdir()
    (project_root / ".env" / "config").write_text("SECRET=1", encoding="utf-8")
    report = client.post("/kb/docs/documents", json={"sources": [str(project_root / ".env")]}).json()
    assert report["failed"] == 1 and report["added"] == 0 and "protégé" in report["errors"][0]

    r = client.request("DELETE", "/kb/docs/documents", json={"source": str(readme)})
    assert r.status_code == 200 and r.json() == {"removed": True}
    r = client.request("DELETE", "/kb/docs/documents", json={"source": str(readme)})
    assert r.json() == {"removed": False}
    r = client.delete("/kb/docs/documents", params={"source": str(project_root / "notes.txt")})
    assert r.status_code == 200 and r.json() == {"removed": True}
    assert client.delete("/kb/docs/documents").status_code == 422
    assert client.get("/kb/docs").json()["n_docs"] == 0


def test_kb_add_folder_respects_deny_patterns(api, settings: Settings, project_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Régression : les fichiers d'un dossier ingéré passent chacun par
    ``check_read`` (motifs de nom ``.env``, ``id_rsa``, ``*.pem`` et de chemin
    ``~/.ssh/**``) ; ils ne sont ni lus, ni indexés, ni cherchables."""
    from dhaos.kb import manager as manager_mod

    client, _ = api
    client.post("/kb", json={"name": "docs"})
    docs = project_root / "docs"
    (docs / ".ssh").mkdir(parents=True)
    (docs / "notes.txt").write_text("Le déploiement passe par systemd.\n", encoding="utf-8")
    (docs / ".env").write_text("SECRET_TOKEN=MARQUEUR_DOTENV\n", encoding="utf-8")
    (docs / ".ssh" / "id_rsa").write_text("-----BEGIN KEY----- MARQUEUR_ID_RSA\n", encoding="utf-8")
    (docs / "cert.pem").write_text("MARQUEUR_PEM\n", encoding="utf-8")
    home = Path(settings.resolve_project_root()).parent / "home"  # HOME du fixture
    (home / ".ssh").mkdir()
    (home / ".ssh" / "id_ed25519").write_text("MARQUEUR_HOME_SSH\n", encoding="utf-8")
    (home / "lisible.txt").write_text("texte lisible\n", encoding="utf-8")

    # Les fichiers refusés ne sont jamais ouverts par le gestionnaire.
    original = manager_mod.KnowledgeManager._ingest_file
    opened: list[str] = []

    def spy(self, base_row, path, report, on_progress):  # type: ignore[no-untyped-def]
        opened.append(str(path))
        return original(self, base_row, path, report, on_progress)

    monkeypatch.setattr(manager_mod.KnowledgeManager, "_ingest_file", spy)

    r = client.post("/kb/docs/documents", json={"sources": [str(docs)], "recursive": True})
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["added"] == 1 and report["failed"] == 3, report
    assert len(report["errors"]) == 3 and all("protégé" in e for e in report["errors"])
    assert opened == [str(docs / "notes.txt")]
    sources = [d["source"] for d in client.get("/kb/docs").json()["documents"]]
    assert sources == [str(docs / "notes.txt")]
    for marker in ("MARQUEUR_ID_RSA", "MARQUEUR_DOTENV", "MARQUEUR_PEM"):
        assert client.post("/kb/search", json={"query": marker, "mode": "keyword"}).json() == []

    # Motif de chemin (~/.ssh/**) : le HOME entier est ingérable, pas la clé.
    report = client.post("/kb/docs/documents", json={"sources": [str(home)], "recursive": True}).json()
    assert report["added"] == 1 and report["failed"] == 1 and "protégé" in report["errors"][0]
    assert client.post("/kb/search", json={"query": "MARQUEUR_HOME_SSH", "mode": "keyword"}).json() == []
    assert str(home / ".ssh" / "id_ed25519") not in opened
    # Non récursif : seuls les fichiers du premier niveau.
    (docs / "sous").mkdir()
    (docs / "sous" / "autre.txt").write_text("profond\n", encoding="utf-8")
    report = client.post("/kb/docs/documents", json={"sources": [str(docs)], "recursive": False}).json()
    assert report["added"] == 0 and report["skipped"] == 1 and report["failed"] == 2


def test_kb_stats(api) -> None:
    client, _ = api
    client.post("/kb", json={"name": "dev"})
    client.post("/kb/dev/notes", json={"text": "une note"})
    r = client.get("/kb/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["bases"] == 1 and body["documents"] == 1 and body["chunks"] >= 1
    assert body["per_base"][0]["name"] == "dev"
    r = client.get("/kb/dev/stats")
    assert r.status_code == 200
    assert r.json()["base"]["name"] == "dev" and r.json()["base"]["documents"] == 1
    assert client.get("/kb/absente/stats").status_code == 404
    health = client.get("/health").json()
    assert health["kb"] == {"bases": 1, "documents": 1, "chunks": body["chunks"]}


# -------------------------------------------------------------------- chat


def test_chat_non_stream_creates_and_reuses_session(api, settings: Settings) -> None:
    client, factory = api
    factory.responses.append("Et re-bonjour.")
    r = client.post("/chat", json={"message": "salut", "stream": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "Bonjour !"
    assert body["stop_reason"] == "end_turn"
    assert body["iterations"] == 0 and body["tool_calls"] == 0 and body["error"] is None
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}
    session_id = body["session_id"]
    assert session_id

    # Le backend a reçu le prompt système et des outils.
    first = factory.backends[0]
    assert first.calls and first.calls[0]["system"]
    assert any(t.name == "kb_search" for t in first.calls[0]["tools"])

    r = client.post("/chat", json={"message": "encore", "session_id": session_id, "stream": False})
    assert r.status_code == 200
    assert r.json()["session_id"] == session_id
    assert r.json()["text"] == "Et re-bonjour."
    # L'historique complet est rejoué au second tour.
    second = factory.backends[1]
    assert [m.role for m in second.calls[0]["messages"]] == ["user", "assistant", "user"]

    session = SessionStore(settings).get(session_id)
    assert [m.role for m in session.messages] == ["user", "assistant", "user", "assistant"]
    assert session.meta["title"] == "salut"
    assert session.meta["backend"] == "fake" and session.meta["model"] == "fake-model"


def test_chat_no_tools_and_backend_selection(api) -> None:
    client, factory = api
    r = client.post(
        "/chat",
        json={"message": "sans outils", "stream": False, "no_tools": True, "backend": "claude", "model": "m-x"},
    )
    assert r.status_code == 200
    assert factory.calls[-1] == ("claude", "m-x")
    assert factory.backends[-1].calls[0]["tools"] == []


def test_chat_session_errors(api) -> None:
    client, _ = api
    assert client.post("/chat", json={"message": "x", "session_id": "absente-0000", "stream": False}).status_code == 404
    r = client.post("/chat", json={"message": "x", "session_id": "../../etc", "stream": False})
    assert r.status_code == 400
    assert client.post("/chat", json={"message": "", "stream": False}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_unknown_backend(settings: Settings) -> None:
    from dhaos.api.server import create_app

    def factory(settings: Settings, name: str | None, model: str | None) -> Backend:
        if name not in (None, "fake"):
            raise ValueError(f"backend inconnu : {name!r}")
        return FakeBackend(["ok"])

    with local_client(create_app(settings, backend_factory=factory, kb_factory=KnowledgeManager)) as client:
        r = client.post("/chat", json={"message": "x", "backend": "nope", "stream": False})
        assert r.status_code == 400
        assert "backend" in r.json()["detail"]
        assert client.get("/sessions").json() == []  # aucune session créée
        assert client.post("/chat", json={"message": "x", "backend": "fake", "stream": False}).status_code == 200


def test_chat_stream_text_then_done(api) -> None:
    client, _ = api
    r = client.post("/chat", json={"message": "salut"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["text", "done"]
    assert events[0][1] == {"text": "Bonjour !"}
    done = events[1][1]
    assert done["session_id"] == r.headers["x-session-id"]
    assert done["stop_reason"] == "end_turn"
    assert done["error"] is None
    assert done["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert done["iterations"] == 0 and done["tool_calls"] == 0
    assert client.get(f"/sessions/{done['session_id']}").status_code == 200


def test_chat_stream_tool_round(settings: Settings) -> None:
    responses: list[str | ChatResponse] = [
        tool_response(tool_call("kb_list", "call_7"), text="Je regarde les bases."),
        "Aucune base pour l'instant.",
    ]
    client, factory = make_client(settings, responses)
    with client:
        client.post("/kb", json={"name": "dev", "description": "base développeur"})
        r = client.post("/chat", json={"message": "quelles bases ?", "stream": True})
        assert r.status_code == 200
        events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds == ["text", "tool_call", "tool_result", "text", "done"]
    assert events[0][1]["text"] == "Je regarde les bases."
    assert events[1][1] == {"id": "call_7", "name": "kb_list", "arguments": {}}
    result = events[2][1]
    assert result["id"] == "call_7" and result["name"] == "kb_list"
    assert result["is_error"] is False
    assert "dev" in result["preview"] and len(result["preview"]) <= 300
    assert events[3][1]["text"] == "Aucune base pour l'instant."
    done = events[4][1]
    assert done["tool_calls"] == 1 and done["iterations"] == 1
    assert done["usage"] == {"input_tokens": 10, "output_tokens": 5}
    # Le résultat d'outil a bien été renvoyé au modèle.
    roles = [m.role for m in factory.backends[0].calls[1]["messages"]]
    assert roles == ["user", "assistant", "tool"]


def test_chat_stream_error_event(settings: Settings) -> None:
    class Crashing(Backend):
        name = "crash"
        model = "none"

        def chat(self, messages, *, system="", tools=None, on_text=None, on_thinking=None) -> ChatResponse:
            raise RuntimeError("panne interne")

    client, _ = make_client(settings, backend_factory=lambda s, n, m: Crashing())
    with client:
        r = client.post("/chat", json={"message": "x"})
        assert r.status_code == 200
        events = parse_sse(r.text)
        assert [e for e, _ in events] == ["error"]
        assert "panne interne" in events[0][1]["detail"]
        # Non-stream : erreur JSON 500.
        r = client.post("/chat", json={"message": "x", "stream": False})
        assert r.status_code == 500
        assert "panne interne" in r.json()["detail"]


def test_chat_backend_error_is_reported_not_raised(settings: Settings) -> None:
    class Down(Backend):
        name = "down"
        model = "none"

        def chat(self, messages, *, system="", tools=None, on_text=None, on_thinking=None) -> ChatResponse:
            raise BackendError("connexion refusée")

    client, _ = make_client(settings, backend_factory=lambda s, n, m: Down())
    with client:
        r = client.post("/chat", json={"message": "x", "stream": False})
        assert r.status_code == 200
        body = r.json()
        assert body["stop_reason"] == "error" and "refusée" in body["error"] and body["text"] == ""
        r = client.post("/chat", json={"message": "x"})
        events = parse_sse(r.text)
        assert [e for e, _ in events] == ["done"]
        assert events[0][1]["stop_reason"] == "error"


def test_chat_session_busy_conflict(api, settings: Settings) -> None:
    client, _ = api
    session = SessionStore(settings).create()
    locks = client.app.state.session_locks
    assert locks.acquire(session.id)
    try:
        r = client.post("/chat", json={"message": "x", "session_id": session.id, "stream": False})
        assert r.status_code == 409
    finally:
        locks.release(session.id)
    assert not locks.is_busy(session.id)
    r = client.post("/chat", json={"message": "x", "session_id": session.id, "stream": False})
    assert r.status_code == 200
    assert not locks.is_busy(session.id)


def _spy_build_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Remplace ``build_runtime`` du module API par un espion qui note ses
    arguments nommés puis délègue à l'original."""
    from dhaos.api import server as api_server

    captured: dict[str, Any] = {}
    original = api_server.build_runtime

    def spy(settings: Settings, **kw: Any):
        captured.update(kw)
        return original(settings, **kw)

    monkeypatch.setattr(api_server, "build_runtime", spy)
    return captured


def test_chat_never_confirms_by_default(api, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from dhaos.policy import never_confirm

    client, _ = api
    captured = _spy_build_runtime(monkeypatch)
    assert client.post("/chat", json={"message": "x", "stream": False}).status_code == 200
    assert settings.api.auto_confirm is False
    assert captured["confirm"] is never_confirm
    assert captured["kb"] is client.app.state.kb
    assert captured["tools"] is True
    assert captured["session"].id == client.get("/sessions").json()[0]["id"]


def test_chat_failure_before_run_leaves_no_session(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Régression : si ``build_runtime`` échoue, la session créée pour ce tour
    (vide) est supprimée ; aucun fichier JSONL orphelin, ``/sessions`` vide."""
    from dhaos.api import server as api_server

    def broken(settings: Settings, **kw: Any) -> Any:
        raise RuntimeError("assemblage impossible")

    monkeypatch.setattr(api_server, "build_runtime", broken)
    client, _ = make_client(settings, ["ok"], raise_server_exceptions=False)
    with client:
        r = client.post("/chat", json={"message": "x", "stream": False})
        assert r.status_code == 500 and "assemblage" in r.json()["detail"]
        r = client.post("/chat", json={"message": "x", "stream": True})
        assert r.status_code == 500
        assert list(Path(settings.sessions_dir).glob("*.jsonl")) == []
        assert client.get("/sessions").json() == []
        assert not client.app.state.session_locks._busy
        # Une session existante n'est jamais supprimée, même vide.
        existing = SessionStore(settings).create()
        r = client.post("/chat", json={"message": "x", "stream": False, "session_id": existing.id})
        assert r.status_code == 500
        assert [i["id"] for i in client.get("/sessions").json()] == [existing.id]


def test_chat_crash_leaves_no_empty_session(settings: Settings) -> None:
    """Régression : un backend qui lève (hors ``BackendError``) ne laisse pas
    de session vide (``n_messages == 0``) dans le magasin."""

    class Crashing(Backend):
        name = "crash"
        model = "none"

        def chat(self, messages, *, system="", tools=None, on_text=None, on_thinking=None) -> ChatResponse:
            raise ValueError("panne")

    client, _ = make_client(settings, backend_factory=lambda s, n, m: Crashing(), raise_server_exceptions=False)
    with client:
        assert client.post("/chat", json={"message": "x", "stream": False}).status_code == 500
        events = parse_sse(client.post("/chat", json={"message": "y"}).text)
        assert [e for e, _ in events] == ["error"]
        infos = client.get("/sessions").json()
        assert all(i["n_messages"] >= 1 for i in infos)
        files = list(Path(settings.sessions_dir).glob("*.jsonl"))
        assert len(files) == len(infos)


def test_chat_agent_crash_before_any_message_leaves_no_session(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Régression : ``Agent.run`` qui lève avant d'enregistrer le moindre
    message (JSON et SSE) ⇒ la session créée pour ce tour est supprimée."""
    from dhaos.agent.loop import Agent

    def boom(self, message, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("panne immédiate")

    monkeypatch.setattr(Agent, "run", boom)
    client, _ = make_client(settings, ["ok"], raise_server_exceptions=False)
    with client:
        assert client.post("/chat", json={"message": "x", "stream": False}).status_code == 500
        r = client.post("/chat", json={"message": "x", "stream": True})
        assert r.status_code == 200
        assert [e for e, _ in parse_sse(r.text)] == ["error"]
        assert client.get("/sessions").json() == []
        assert list(Path(settings.sessions_dir).glob("*.jsonl")) == []
        assert not client.app.state.session_locks._busy


def test_chat_auto_confirm_setting(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from dhaos.policy import auto_confirm

    settings.api.auto_confirm = True
    captured = _spy_build_runtime(monkeypatch)
    client, _ = make_client(settings, ["ok"])
    with client:
        assert client.post("/chat", json={"message": "x", "stream": False}).status_code == 200
    assert captured["confirm"] is auto_confirm


# ---------------------------------------------------------------- sessions


def test_sessions_list_get_delete(api, settings: Settings) -> None:
    client, factory = api
    factory.responses.append("Seconde.")
    assert client.get("/sessions").json() == []
    sid1 = client.post("/chat", json={"message": "première question", "stream": False}).json()["session_id"]
    sid2 = client.post("/chat", json={"message": "seconde question", "stream": False}).json()["session_id"]

    r = client.get("/sessions")
    assert r.status_code == 200
    infos = r.json()
    assert {i["id"] for i in infos} == {sid1, sid2}
    for info in infos:
        assert "path" not in info
        assert set(info) == {"id", "title", "backend", "model", "created_at", "updated_at", "n_messages"}
        assert info["n_messages"] == 2 and info["backend"] == "fake"
    assert {i["title"] for i in infos} == {"première question", "seconde question"}

    r = client.get(f"/sessions/{sid1}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["id"] == sid1
    assert detail["meta"]["title"] == "première question"
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "Bonjour !"
    assert detail["events"] and detail["events"][0]["kind"] == "usage"

    assert client.get("/sessions/absente-0000").status_code == 404
    assert client.get("/sessions/bad%20id!").status_code == 400
    assert client.delete("/sessions/bad%20id!").status_code == 400
    assert client.get("/sessions/..%2F..%2Fetc").status_code in (400, 404)  # jamais de traversée
    assert client.delete(f"/sessions/{sid1}").status_code == 204
    assert client.delete(f"/sessions/{sid1}").status_code == 404
    assert client.get(f"/sessions/{sid1}").status_code == 404
    assert [i["id"] for i in client.get("/sessions").json()] == [sid2]
    assert not (settings.sessions_dir / f"{sid1}.jsonl").exists()


# ----------------------------------------------------------------- journal


def test_journal_tail(api, settings: Settings) -> None:
    client, _ = api
    assert client.get("/journal").json() == []
    journal = Journal(settings.journal_path)
    for i in range(5):
        journal.record("write", path=f"/tmp/f{i}", bytes=i)
    r = client.get("/journal")
    assert r.status_code == 200
    entries = r.json()
    assert len(entries) == 5 and entries[-1]["path"] == "/tmp/f4"
    r = client.get("/journal", params={"n": 2})
    assert [e["path"] for e in r.json()] == ["/tmp/f3", "/tmp/f4"]
    assert client.get("/journal", params={"n": 0}).status_code == 422
    assert client.get("/journal", params={"n": "abc"}).status_code == 422


# ------------------------------------------------------------- unitaires


def test_sse_encoding_and_parsing() -> None:
    from dhaos.api.server import sse

    raw = sse("text", {"text": "ligne 1\nligne 2 — é"})
    assert raw.startswith(b"event: text\ndata: ")
    assert raw.endswith(b"\n\n")
    assert raw.count(b"\n") == 3  # aucun retour à la ligne brut dans data
    assert parse_sse(raw.decode("utf-8")) == [("text", {"text": "ligne 1\nligne 2 — é"})]


def test_masked_config_helper(settings: Settings) -> None:
    from dhaos.api.server import masked_config

    settings.api.token = "abc"
    data = masked_config(settings)
    assert data["api"]["token"] == "***"
    assert data["web"]["brave_api_key"] is None
    assert settings.api.token == "abc"  # l'objet n'est pas modifié


def test_default_factories(settings: Settings) -> None:
    from dhaos.api.server import default_backend_factory, default_kb_factory

    backend = default_backend_factory(settings, "ollama", "modele-x")
    assert backend.name == "ollama" and backend.model == "modele-x"
    with pytest.raises(ValueError):
        default_backend_factory(settings, "inconnu", None)
    kb = default_kb_factory(settings)
    try:
        assert isinstance(kb, KnowledgeManager)
    finally:
        kb.close()


def test_default_kb_factory_used_when_none(settings: Settings) -> None:
    from dhaos.api.server import create_app

    app = create_app(settings, backend_factory=lambda s, n, m: FakeBackend(["ok"]))
    with local_client(app) as client:
        assert isinstance(client.app.state.kb, KnowledgeManager)
        assert client.get("/kb").json() == []


# ------------------------------------------------------------ interface web
def test_ui_page_and_assets_served_without_token(api) -> None:
    client, _ = api
    r = client.get("/", headers=without_auth(client))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "dhaos" in r.text and "/ui/app.js" in r.text and "/ui/style.css" in r.text
    js = client.get("/ui/app.js", headers=without_auth(client))
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "/chat/confirm" in js.text
    css = client.get("/ui/style.css", headers=without_auth(client))
    assert css.status_code == 200 and "css" in css.headers["content-type"]
    assert client.get("/ui/../pyproject.toml", headers=without_auth(client)).status_code == 404


def test_same_origin_requests_allowed_foreign_origin_refused(api) -> None:
    """La page servie par l'API envoie ``Origin`` = sa propre adresse : accepté ;
    une autre origine reste refusée (rebinding, site tiers)."""
    client, _ = api
    assert client.get("/kb", headers={"Origin": "http://127.0.0.1"}).status_code == 200
    assert client.post("/kb/search", json={"query": "x"}, headers={"Origin": "http://127.0.0.1"}).status_code == 200
    r = client.get("/kb", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_models_endpoint(api, settings: Settings) -> None:
    client, _ = api
    claude = client.get("/models", params={"backend": "claude"}).json()
    assert claude["backend"] == "claude" and "claude-opus-5" in claude["models"] and claude["default"] == settings.backends.claude.model
    fake = client.get("/models", params={"backend": "ollama"}).json()  # la fabrique renvoie un FakeBackend
    assert fake["ok"] is True and fake["default"] == "fake-model" and fake["models"] == []
    assert client.get("/models", params={"backend": "inconnu"}).json()["ok"] is False


def test_kb_reindex_endpoint(api) -> None:
    client, _ = api
    client.post("/kb", json={"name": "dev"})
    client.post("/kb/dev/notes", json={"text": "Le port du serveur est 8443.", "title": "port"})
    r = client.post("/kb/dev/reindex")
    assert r.status_code == 200 and r.json()["chunks"] >= 1
    assert client.post("/kb/absente/reindex").status_code == 404


def test_config_patch_writes_file_and_applies_live(api, settings: Settings, tmp_path: Path) -> None:
    client, _ = api
    cfg = client.get("/config").json()
    assert cfg["source_path"] and cfg["project_root"] == str(settings.resolve_project_root())
    r = client.patch("/config", json={"key": "backends.default", "value": "claude"})
    assert r.status_code == 200, r.text
    assert settings.backends.default == "claude"  # appliqué à chaud
    text = Path(r.json()["path"]).read_text(encoding="utf-8")
    assert 'default = "claude"' in text
    r = client.patch("/config", json={"key": "agent.auto_kb_search", "value": False})
    assert r.status_code == 200 and settings.agent.auto_kb_search is False
    assert 'auto_kb_search = false' in Path(r.json()["path"]).read_text(encoding="utf-8")
    assert client.patch("/config", json={"key": "tools.write_policy", "value": "bogus"}).status_code == 400
    assert client.patch("/config", json={"key": "api.token", "value": "x"}).status_code == 400
    assert client.patch("/config", json={"key": "tools.deny_patterns", "value": []}).status_code == 400
    assert settings.tools.write_policy == "project"


@contextlib.contextmanager
def live_server(app: Any) -> Iterator[str]:
    """Vrai serveur uvicorn dans un thread (le ``TestClient`` sérialise les
    requêtes et ne délivre pas un flux SSE au fil de l'eau) ; renvoie l'URL."""
    import socket
    import threading
    import time

    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "serveur uvicorn non démarré"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _stream_with_confirm(base_url: str, token: str, payload: dict[str, Any], answer: bool | None) -> list[tuple[str, Any]]:
    """Lit le flux SSE d'un vrai serveur et, sur ``confirm``, répond via
    ``POST /chat/confirm`` pendant que le flux est ouvert."""
    import httpx

    events: list[tuple[str, Any]] = []
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=30.0) as client:
        with client.stream("POST", "/chat", json=payload) as r:
            assert r.status_code == 200
            buffer = ""
            for chunk in r.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    for event, data in parse_sse(block + "\n\n"):
                        events.append((event, data))
                        if event == "confirm" and answer is not None:
                            reply = client.post("/chat/confirm", json={"id": data["id"], "answer": answer})
                            assert reply.status_code == 200, reply.text
    return events


@pytest.mark.parametrize("answer", [True, False])
def test_chat_stream_confirmation_flow(settings: Settings, project_root: Path, answer: bool) -> None:
    """Commande hors liste blanche : l'agent demande confirmation dans le flux
    (``event: confirm``) ; ``oui`` exécute, ``non`` refuse."""
    from dhaos.api.server import create_app

    settings.tools.shell_policy = "ask"
    settings.api.confirm_timeout = 10.0
    settings.api.token = "t-confirm"
    responses: list[str | ChatResponse] = [
        tool_response(tool_call("run_command", "c1", command="touch confirme.txt")),
        "Terminé.",
    ]
    app = create_app(settings, backend_factory=BackendFactory(responses), kb_factory=KnowledgeManager)
    with live_server(app) as base_url:
        events = _stream_with_confirm(base_url, "t-confirm", {"message": "crée le fichier", "stream": True}, answer)
    kinds = [e for e, _ in events]
    assert kinds == ["tool_call", "confirm", "tool_result", "text", "done"], kinds
    confirm = events[1][1]
    assert confirm["prompt"].startswith("Exécuter : touch confirme.txt") and confirm["id"]
    result = events[2][1]
    assert result["is_error"] is (not answer)
    assert (project_root / "confirme.txt").exists() is answer
    if not answer:
        assert "refus" in result["preview"]


def test_chat_stream_confirmation_timeout_refuses(settings: Settings, project_root: Path) -> None:
    from dhaos.api.server import create_app

    settings.tools.shell_policy = "ask"
    settings.api.confirm_timeout = 1.0
    settings.api.token = "t-timeout"
    app = create_app(settings, backend_factory=BackendFactory([tool_response(tool_call("run_command", "c1", command="touch tard.txt")), "ok"]), kb_factory=KnowledgeManager)
    with live_server(app) as base_url:
        events = _stream_with_confirm(base_url, "t-timeout", {"message": "x", "stream": True}, None)
    kinds = [e for e, _ in events]
    assert kinds == ["tool_call", "confirm", "tool_result", "text", "done"]
    assert events[2][1]["is_error"] is True and not (project_root / "tard.txt").exists()
    assert app.state.confirm_broker.pending() == 0


def test_chat_confirm_unknown_id_is_404(api) -> None:
    client, _ = api
    assert client.post("/chat/confirm", json={"id": "nope", "answer": True}).status_code == 404


def test_chat_non_stream_does_not_wait_for_confirmation(settings: Settings, project_root: Path) -> None:
    """Sans flux, personne ne peut répondre : la politique configurée s'applique (refus)."""
    settings.tools.shell_policy = "ask"
    client, _ = make_client(settings, [tool_response(tool_call("run_command", "c1", command="touch nonstream.txt")), "ok"])
    with client:
        r = client.post("/chat", json={"message": "x", "stream": False})
    assert r.status_code == 200 and r.json()["tool_calls"] == 1
    assert not (project_root / "nonstream.txt").exists()


def test_chat_confirm_endpoint_answers_pending_request(api) -> None:
    """``POST /chat/confirm`` débloque une demande en attente (le fil de l'agent
    est simulé par un thread qui appelle ``ConfirmBroker.ask``)."""
    import queue
    import threading

    client, _ = api
    broker = client.app.state.confirm_broker
    events: "queue.Queue[Any]" = queue.Queue()
    outcome: dict[str, Any] = {}

    def agent_side() -> None:
        outcome["answer"] = broker.ask("Exécuter : rm -rf build ? ", "s1", events)

    t = threading.Thread(target=agent_side, daemon=True)
    t.start()
    event, data = events.get(timeout=5)
    assert event == "confirm" and data["prompt"].startswith("Exécuter") and data["session_id"] == "s1"
    r = client.post("/chat/confirm", json={"id": data["id"], "answer": True})
    assert r.status_code == 200 and r.json()["accepted"] is True
    t.join(timeout=5)
    assert outcome["answer"] is True and broker.pending() == 0
    assert client.post("/chat/confirm", json={"id": data["id"], "answer": True}).status_code == 404
