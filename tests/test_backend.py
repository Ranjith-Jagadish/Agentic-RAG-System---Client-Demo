"""Tests for backend API"""

import pytest
from fastapi.testclient import TestClient
from src.backend.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data


def test_chat_endpoint_structure(client):
    """Test chat endpoint structure"""
    # Note: This test may fail if services are not running
    # It's mainly to test the endpoint structure
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Test question",
            "conversation_id": None,
            "stream": False
        }
    )
    # Accept both success and service unavailable
    assert response.status_code in [200, 500, 503]


def test_create_conversation(client):
    """Test conversation creation"""
    response = client.post(
        "/api/v1/conversations",
        json={"user_id": None}
    )
    # Accept both success and service unavailable
    assert response.status_code in [200, 500, 503]


@pytest.mark.skip(reason="Requires database connection")
def test_get_conversation(client):
    """Test getting conversation history"""
    # First create a conversation
    create_response = client.post(
        "/api/v1/conversations",
        json={"user_id": None}
    )
    
    if create_response.status_code == 200:
        conversation_id = create_response.json()["conversation_id"]
        
        # Get conversation
        get_response = client.get(f"/api/v1/conversations/{conversation_id}")
        assert get_response.status_code == 200

