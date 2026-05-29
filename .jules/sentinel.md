## 2026-05-26 - Content-Security-Policy Implementation for Legacy UI
**Vulnerability:** XSS and Clickjacking risks due to missing security headers.
**Learning:** Adding a Content-Security-Policy to an existing application requires balancing security and functionality. The current UI relies on jQuery and potentially inline scripts/styles, necessitating `'unsafe-inline'` for immediate compatibility.
**Prevention:** In future development, move all inline scripts and styles to external files and avoid CDNs or use Subresource Integrity (SRI) to allow for a stricter CSP.

## 2026-05-29 - Defense in Depth: XSS and Input Validation
**Vulnerability:** DOM-based XSS via `innerHTML` and missing backend input validation.
**Learning:** Security headers (like CSP) are a great layer of defense but do not replace the need for secure coding practices like using `textContent` and strict server-side validation. Accidental environment pollution (e.g., committing test artifacts) can compromise repository health.
**Prevention:** Always use `textContent` for user-provided data in the frontend. Implement strict allowlist and length validation on the backend. Use `fastapi.testclient.TestClient` for faster, cleaner security regression tests.
