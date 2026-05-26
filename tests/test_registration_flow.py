import pytest
from playwright.sync_api import sync_playwright
import time
import subprocess
import os
import json

@pytest.fixture(scope="module")
def server():
    # Start the server in mock mode
    env = os.environ.copy()
    env["MOCK_MODE"] = "true"
    process = subprocess.Popen(["python3", "main.py"], env=env)
    time.sleep(15)
    yield
    process.terminate()

def test_registration_flow(server):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Mock /unknown_faces to return one face
        page.route("**/unknown_faces", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps([{"id": "test-face-id", "image_url": "/static/mock_face.jpg"}])
        ))

        # Mock /users/add to return success
        page.route("**/users/add", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "success", "user_id": 1})
        ))

        # Mock /users to return the new user after addition
        # Initially empty, then with user
        users_count = [0]
        def handle_users(route):
            if users_count[0] == 0:
                route.fulfill(status=200, body=json.dumps([]))
                users_count[0] += 1
            else:
                route.fulfill(status=200, body=json.dumps([
                    {"id": 1, "name": "Test User", "status": "approved", "image_path": "faces/test.jpg"}
                ]))

        page.route("**/users", handle_users)

        page.goto("http://localhost:8000")

        # Wait for the unknown face to appear
        page.wait_for_selector(".unknown-item")

        # Click the unknown face
        page.click(".unknown-item")

        # Check if modal is visible
        assert page.is_visible("#registration-modal")

        # Fill the form
        page.fill("#reg-name", "Test User")
        page.select_option("#reg-status", "approved")

        # Submit the form
        page.click("#registration-form button[type='submit']")

        # Wait for modal to close
        page.wait_for_selector("#registration-modal", state="hidden")

        # Verify the user appears in the list
        # Note: the polling might take a few seconds
        page.wait_for_selector("#user-list-body tr")
        user_row = page.locator("#user-list-body tr")
        assert "Test User" in user_row.inner_text()
        assert "approved" in user_row.inner_text()

        browser.close()
