"""SSRF-safe URL fetcher that strictly allows only public addresses and direct media."""

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

MAX_REDIRECTS = 3
FETCH_TIMEOUT = 20.0


def is_public_ip(ip_str: str) -> bool:
    """Verify that an IP address is a public, globally routable address."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return False


def validate_target_url(url: str) -> tuple[str, str, int]:
    """Validate URL scheme and port, and resolve IP to verify it is public.

    Returns (parsed_hostname, resolved_ip, port).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https schemes are permitted.")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in (80, 443):
        raise ValueError("Only standard ports 80 and 443 are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: missing hostname.")

    # Resolve IP address
    try:
        addrinfo = socket.getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: {hostname}") from exc

    resolved_ips = [item[4][0] for item in addrinfo]
    for ip in resolved_ips:
        if not is_public_ip(ip):
            raise ValueError(f"Disallowed destination address: {ip} is private or non-routable.")

    return hostname, resolved_ips[0], port


def download_media_from_url(url: str, max_bytes: int) -> tuple[bytes, str]:
    """Download media bytes from URL with SSRF checks, redirect limits, and size caps.

    Returns (downloaded_bytes, final_url).
    """
    current_url = url
    redirects = 0

    while True:
        validate_target_url(current_url)

        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=False) as client:
            try:
                response = client.get(current_url, headers={"User-Agent": "TruthLens-Verifier/1.0"})
            except httpx.RequestError as exc:
                raise ValueError(f"Failed to fetch URL: {exc}") from exc

            # Handle redirects manually to enforce SSRF validation at every hop
            if response.is_redirect:
                redirects += 1
                if redirects > MAX_REDIRECTS:
                    raise ValueError(f"Too many redirects (maximum {MAX_REDIRECTS}).")
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Redirect response missing Location header.")
                # Resolve relative URL if necessary
                current_url = str(response.url.join(location))
                continue

            if response.status_code != 200:
                raise ValueError(f"Remote server returned HTTP {response.status_code}.")

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"File exceeds maximum allowed size of {max_bytes} bytes.")

            content = response.content
            if len(content) > max_bytes:
                raise ValueError(f"Downloaded content exceeds maximum allowed size of {max_bytes} bytes.")

            return content, current_url
