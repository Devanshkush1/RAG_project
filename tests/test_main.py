import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_upload_documents():
    with open("articles/sample_article.txt", "rb") as f:
        response = client.post("/upload-documents", files=[("files", ("sample_article.txt", f, "text/plain"))])
    assert response.status_code == 200
    assert response.json() == {"message": "Documents uploaded successfully"}

def test_query():
    response = client.post("/query", json={"query": "What is AI?"})
    assert response.status_code == 200
    assert "response" in response.json()

def test_get_metadata():
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)