import re
import json
from typing import Any, Dict, Optional

import requests
from langchain_core.tools import tool

# External URLs (company policies / help center)
URL_POLICY = "https://www.palaceresorts.com/terms-conditions-website-usage"
URL_QA = "https://www.palaceresorts.com/help-center/bookings-and-cancellations"

# Base URL for internal API (pricing, stats, health, etc.)
BASE_URL = "https://office-hours-buildathon.palaceresorts.com/api"


def _safe_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 8) -> Optional[Dict[str, Any]]:
    """Perform a GET request and return parsed JSON or None on error."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"text": resp.text}
    except Exception as e:
        return {"error": str(e)}


def _safe_post(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Perform a POST request with JSON payload and return parsed JSON or error dict."""
    try:
        resp = requests.post(url, json=payload or {}, timeout=timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"text": resp.text}
    except Exception as e:
        return {"error": str(e)}


@tool
def consult_policy(query: str) -> str:
    """Fetch Palace Resorts policy page and return a short excerpt relevant to the query.
    
    This tool performs a simple keyword search on the fetched text and returns the
    matching sentences. If fetching the page fails, it returns the policy URL.
    """
    result = _safe_get(URL_POLICY)
    if not result:
        return f"Could not fetch policy page. Please consult: {URL_POLICY}"
    
    # the API may return {'text': html} or a JSON; handle both
    text = result.get("text") if isinstance(result, dict) else result
    if isinstance(text, dict):
        text = json.dumps(text)
    
    if not text:
        return f"Policy page empty; consult: {URL_POLICY}"
    
    # Naive text search: find sentences containing query tokens
    query_tokens = re.findall(r"\w+", query.lower())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matches = []
    for s in sentences:
        sl = s.lower()
        if all(any(tok in sl for tok in [t]) for t in query_tokens[:3]):
            matches.append(s.strip())
        if len(matches) >= 5:
            break
    
    if matches:
        return "\n\n".join(matches)
    # Fallback: return short excerpt
    excerpt = text[:2000]
    return f"Policy page (excerpt):\n\n{excerpt}\n\nFull policy: {URL_POLICY}"


@tool
def frequent_questions(query: str) -> str:
    """Fetch the FAQ / Help Center page and return an excerpt or matching Q&A.
    
    The function performs a simple keyword search on the page text and returns
    the most relevant snippets. If unavailable, it returns the help center URL.
    """
    result = _safe_get(URL_QA)
    if not result:
        return f"Could not fetch FAQ page. Please consult: {URL_QA}"
    
    text = result.get("text") if isinstance(result, dict) else result
    if isinstance(text, dict):
        text = json.dumps(text)
    
    if not text:
        return f"FAQ page empty; consult: {URL_QA}"
    
    query_tokens = re.findall(r"\w+", query.lower())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    matches = [s.strip() for s in sentences if any(tok in s.lower() for tok in query_tokens)]
    if matches:
        return "\n\n".join(matches[:8])
    
    excerpt = text[:2000]
    return f"FAQ page (excerpt):\n\n{excerpt}\n\nFull FAQ: {URL_QA}"


@tool
def db_get(path: str, payload: Optional[Dict[str, Any]] = None, method: str = "get") -> Dict[str, Any]:
    """Generic API wrapper to query the backend system.
    
    Args:
        path: endpoint path relative to BASE_URL, e.g. 'pricing' or '/stats'
        payload: JSON payload for POST requests
        method: 'get' or 'post'
    
    Returns parsed JSON or an error dict.
    """
    # Normalize path
    if path.startswith("/"):
        path = path[1:]
    url = f"{BASE_URL.rstrip('/')}/{path}"
    
    if method.lower() == "post":
        return _safe_post(url, payload)
    else:
        return _safe_get(url, params=payload)


@tool
def check_server_health() -> Dict[str, Any]:
    """Check root server health (non-API subpath)."""
    health_url = BASE_URL.replace("/api", "") + "/health"
    return _safe_get(health_url)


@tool
def get_stats() -> Dict[str, Any]:
    """Return global statistics from the backend API."""
    return _safe_get(f"{BASE_URL}/stats")


@tool
def calculate_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call the pricing endpoint with the given payload and return parsed JSON."""
    return _safe_post(f"{BASE_URL}/pricing", payload)


@tool
def get_booking_channels() -> Dict[str, Any]:
    """Fetch booking channels from the backend API."""
    return _safe_get(f"{BASE_URL}/booking_channels")


@tool
def get_customer_segments() -> Dict[str, Any]:
    """Fetch customer segments from the backend API."""
    return _safe_get(f"{BASE_URL}/customer_segments")


@tool
def get_properties() -> Dict[str, Any]:
    """Fetch properties information from the backend API."""
    return _safe_get(f"{BASE_URL}/properties")


@tool
def get_reservations() -> Dict[str, Any]:
    """Fetch reservations data from the backend API."""
    return _safe_get(f"{BASE_URL}/reservations")


@tool
def get_room_types() -> Dict[str, Any]:
    """Fetch room types catalog from the backend API."""
    return _safe_get(f"{BASE_URL}/room_types_catalog")


@tool
def get_rooms_inventory() -> Dict[str, Any]:
    """Fetch rooms inventory from the backend API."""
    return _safe_get(f"{BASE_URL}/rooms_inventory")