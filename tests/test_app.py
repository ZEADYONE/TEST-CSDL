from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app


class FakeSession:
    def execute(self, _statement):
        return 1

    def scalar(self, _statement):
        return 0


def fake_db():
    yield FakeSession()


def test_health_and_home_routes():
    app.dependency_overrides[get_db] = fake_db
    try:
        with TestClient(app) as client:
            health = client.get("/api/health")
            home = client.get("/")
        assert health.status_code == 200
        assert health.json()["feature_version"] == "v5-bbox-shape"
        assert home.status_code == 200
        assert "Bird Image Retrieval" in home.text
        assert "same_species" not in home.text
    finally:
        app.dependency_overrides.clear()
