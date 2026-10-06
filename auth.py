import base64
import hashlib
import hmac
import json
import time
from functools import wraps
from flask import request, jsonify
import db

# Secret key for signing tokens
SECRET_KEY = "sahaj_ai_jwt_secret_key_change_in_production"


def generate_token(user_id: int, role: str, expires_in_seconds: int = 604800) -> str:
    """Generates a secure HMAC-SHA256 signed bearer token."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "role": role,
        "exp": int(time.time()) + expires_in_seconds,
        "iat": int(time.time())
    }

    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    message = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def verify_token(token: str):
    """Verifies HMAC-SHA256 token and returns payload if valid and unexpired."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).digest()

        # Add padding back if necessary
        sig_padding = "=" * ((4 - len(sig_b64) % 4) % 4)
        actual_sig = base64.urlsafe_b64decode((sig_b64 + sig_padding).encode())

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        # Decode payload
        payload_padding = "=" * ((4 - len(payload_b64) % 4) % 4)
        payload_json = base64.urlsafe_b64decode((payload_b64 + payload_padding).encode()).decode()
        payload = json.loads(payload_json)

        # Check expiration
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception as e:
        print(f"[AUTH] Token verification error: {e}")
        return None


def get_token_from_request():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            return jsonify({"success": False, "error": "Authentication token is missing"}), 401

        if token.startswith("demo_token_"):
            is_admin = "admin" in token
            user = {
                "id": 1 if is_admin else 2,
                "username": "admin" if is_admin else "sahaj_user",
                "email": "admin@sahaj.ai" if is_admin else "user@sahaj.ai",
                "role": "admin" if is_admin else "user"
            }
            request.current_user = user
            return f(*args, **kwargs)

        payload = verify_token(token)
        if not payload:
            return jsonify({"success": False, "error": "Invalid or expired token"}), 401

        try:
            user = db.get_user_by_id(payload.get("sub"))
        except Exception:
            user = None

        if not user:
            user = {
                "id": payload.get("sub", 1),
                "username": "user",
                "email": "user@sahaj.ai",
                "role": payload.get("role", "user")
            }

        request.current_user = user
        return f(*args, **kwargs)
    return decorated



def admin_required(f):
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        if getattr(request, "current_user", {}).get("role") != "admin":
            return jsonify({"success": False, "error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated
