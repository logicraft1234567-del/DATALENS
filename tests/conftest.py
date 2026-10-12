import importlib
import io
import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SALES_CSV = """order_id,product,region,revenue,units,order_date,email
1,Widget,North,100.5,2,2024-01-05,a@example.com
2,Widget,South,200.0,4,2024-01-20,b@example.com
3,Gadget,North,50.25,1,2024-02-03,c@example.com
4,Gadget,South,75.0,3,2024-02-18,d@example.com
5,Doohickey,North,300.0,6,2024-03-02,e@example.com
6,Doohickey,East,120.0,2,2024-03-15,f@example.com
7,Widget,East,90.0,2,2024-04-01,g@example.com
8,Gadget,East,60.0,1,2024-04-12,h@example.com
9,Widget,North,110.0,3,2024-05-09,i@example.com
10,Doohickey,South,310.0,7,2024-05-21,j@example.com
"""


class FakeGroq:
    """Stands in for the Groq call so tests never need credentials or network access."""

    def __init__(self):
        self.calls, self.plan, self.error = [], {"operations": []}, None

    def __call__(self, messages, max_tokens, temperature=0.2, model=None):
        self.calls.append({"messages": messages, "model": model})
        if self.error is not None:
            raise self.error
        if "JSON plan" in messages[0]["content"]:
            return json.dumps(self.plan), 11
        return "Fake answer from the model.", 22

    @property
    def answer_calls(self):
        return [c for c in self.calls if "JSON plan" not in c["messages"][0]["content"]]


class Client:
    """Flask test client that behaves like the browser front end (CSRF token on every state-changing call)."""

    def __init__(self, app):
        self.c = app.test_client()
        self.csrf = self.c.get("/api/me").get_json()["csrf_token"]

    def call(self, method, url, body=None, headers=None, **kw):
        h = {"X-CSRFToken": self.csrf, **(headers or {})}
        resp = getattr(self.c, method)(url, json=body, headers=h, **kw) if body is not None or method in ("post", "patch", "delete") else getattr(self.c, method)(url, headers=h, **kw)
        data = resp.get_json(silent=True)
        if data and data.get("csrf_token"):
            self.csrf = data["csrf_token"]
        return resp

    def get(self, url, **kw): return self.call("get", url, **kw)
    def post(self, url, body=None, **kw): return self.call("post", url, {} if body is None else body, **kw)
    def patch(self, url, body=None, **kw): return self.call("patch", url, {} if body is None else body, **kw)
    def delete(self, url, body=None, **kw): return self.call("delete", url, {} if body is None else body, **kw)


@pytest.fixture(scope="session")
def app_mod():
    tmp = tempfile.mkdtemp(prefix="datalens-tests-")
    os.environ.update({"DATALENS_ENV": "development", "SECRET_KEY": "test-secret", "DATABASE_URL": f"sqlite:///{tmp}/test.db",
                       "UPLOAD_FOLDER": f"{tmp}/uploads", "GROQ_API_KEY": "test-key-not-real", "ASSISTANT_MODELS": "openai/gpt-oss-120b"})
    os.environ.pop("INITIAL_ADMIN_PASSWORD", None)
    sys.modules.pop("app", None)
    mod = importlib.import_module("app")
    mod.limiter.enabled = False
    return mod


@pytest.fixture()
def fake(app_mod, monkeypatch):
    f = FakeGroq()
    monkeypatch.setattr(app_mod, "call_groq", f)
    return f


_counter = {"n": 0}


def make_user(app_mod, prefix="user"):
    _counter["n"] += 1
    name = f"{prefix}{_counter['n']}"
    cl = Client(app_mod.app)
    r = cl.post("/api/auth/signup", {"full_name": "Test Person", "username": name, "email": f"{name}@example.com",
                                     "password": "Password123", "confirm_password": "Password123"})
    assert r.status_code == 201, r.get_json()
    return cl


def upload(cl, csv_text=SALES_CSV, name="sales.csv"):
    r = cl.c.post("/api/datasets", data={"file": (io.BytesIO(csv_text.encode()), name)}, headers={"X-CSRFToken": cl.csrf},
                  content_type="multipart/form-data")
    assert r.status_code == 201, r.get_json()
    return r.get_json()["dataset"]["id"]


@pytest.fixture()
def owner(app_mod):
    return make_user(app_mod, "owner")


@pytest.fixture()
def stranger(app_mod):
    return make_user(app_mod, "stranger")


@pytest.fixture()
def setup_assistant(app_mod, owner, fake):
    """An owner with a dataset and a tested assistant (not yet published)."""
    ds = upload(owner)
    a = owner.post("/api/assistants", {"name": "Sales helper", "dataset_id": ds, "description": "Answers sales questions",
                                      "instructions": "Be friendly."}).get_json()["assistant"]
    conv = owner.post(f"/api/assistants/{a['id']}/conversations").get_json()["conversation"]
    r = owner.post(f"/api/assistant-conversations/{conv['id']}/messages", {"content": "How many orders are there?"})
    assert r.status_code == 200, r.get_json()
    fake.calls.clear()
    return SimpleNS(owner=owner, ds=ds, a=owner.get(f"/api/assistants/{a['id']}").get_json()["assistant"], conv=conv)


class SimpleNS:
    def __init__(self, **kw): self.__dict__.update(kw)


def publish(owner, aid):
    r = owner.post(f"/api/assistants/{aid}/publish", {"acknowledge": True})
    assert r.status_code == 200, r.get_json()
    return r


def new_key(owner, aid, name="server"):
    r = owner.post(f"/api/assistants/{aid}/keys", {"name": name})
    assert r.status_code == 201, r.get_json()
    return r.get_json()


def api_post(app_mod, aid, token, body, raw=None, headers=None):
    # a fresh client with no cookies and no CSRF token: exactly what an external server would send
    h = {"Authorization": f"Bearer {token}", **(headers or {})} if token else dict(headers or {})
    c = app_mod.app.test_client()
    if raw is not None:
        return c.post(f"/api/v1/assistants/{aid}/chat", data=raw, headers=h, content_type="application/json")
    return c.post(f"/api/v1/assistants/{aid}/chat", json=body, headers=h)