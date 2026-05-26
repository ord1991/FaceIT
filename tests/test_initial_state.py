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
        page.goto("http://localhost:8000", timeout=60000)

        # Check if Registered Users table has no rows (tr)
        user_rows = page.locator("#user-list-body tr")
        assert user_rows.count() == 0

        # Check if Unknown Faces container has no children (div)
        unknown_items = page.locator("#unknown-faces-container > div")
        assert unknown_items.count() == 0

        browser.close()
