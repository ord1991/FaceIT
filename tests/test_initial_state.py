import pytest
from playwright.sync_api import sync_playwright
import time
import subprocess
import os

@pytest.fixture(scope="module")
def server():
    # Start the server in mock mode
    env = os.environ.copy()
    env["MOCK_MODE"] = "true"
    process = subprocess.Popen(["python3", "main.py"], env=env)

    # Wait for server to start
    time.sleep(15)
    yield
    process.terminate()

def test_initial_state(server):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Force empty users list via routing to avoid leftover state
        page.route("**/users", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body="[]"
        ))

        page.goto("http://localhost:8000", timeout=60000)

        # Check if Registered Users table has the empty state row
        user_rows = page.locator("#user-list-body tr")
        # Wait for the JS to populate the empty state
        page.wait_for_selector("#user-list-body tr")
        assert user_rows.count() == 1
        assert "No users registered" in user_rows.first.inner_text()

        # Check if Unknown Faces container has no children (button)
        unknown_items = page.locator("#unknown-faces-container > button")
        assert unknown_items.count() == 0

        browser.close()
