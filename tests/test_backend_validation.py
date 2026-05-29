import pytest
from fastapi.testclient import TestClient
from main import app
from database import Base, engine, get_db
from sqlalchemy.orm import sessionmaker

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_faceit.db"
test_engine = engine # We can use the same engine but ideally a different one.
# For simplicity, we just use a separate session and cleanup.

@pytest.fixture
def client():
    # We don't really need to mock the DB for these validation tests
    # as they should fail BEFORE hitting the DB if possible,
    # or at least we can just use the default DB for these simple checks.
    with TestClient(app) as c:
        yield c

def test_add_user_validation(client):
    # Test invalid status
    response = client.post("/users/add", data={"name": "Test", "status": "hacker"})
    assert response.status_code == 400
    assert "Invalid status" in response.json()["detail"]

    # Test empty name
    response = client.post("/users/add", data={"name": "  ", "status": "approved"})
    assert response.status_code == 400
    assert "Invalid name" in response.json()["detail"]

    # Test too long name
    response = client.post("/users/add", data={"name": "A" * 101, "status": "approved"})
    assert response.status_code == 400
    assert "Invalid name" in response.json()["detail"]

def test_update_status_validation(client):
    # Test invalid status on update
    response = client.post("/users/update_status", data={"user_id": 1, "status": "malicious"})
    assert response.status_code == 400
    assert "Invalid status" in response.json()["detail"]
