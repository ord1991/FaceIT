## 2026-05-26 - Content-Security-Policy Implementation for Legacy UI
**Vulnerability:** XSS and Clickjacking risks due to missing security headers.
**Learning:** Adding a Content-Security-Policy to an existing application requires balancing security and functionality. The current UI relies on jQuery and potentially inline scripts/styles, necessitating `'unsafe-inline'` for immediate compatibility.
**Prevention:** In future development, move all inline scripts and styles to external files and avoid CDNs or use Subresource Integrity (SRI) to allow for a stricter CSP.
