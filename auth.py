import os
import secrets
import logging

logger = logging.getLogger(__name__)

ADMIN_TOKEN = None


def ensure_admin_token():
    global ADMIN_TOKEN
    token_file = ".admin"
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            token = f.read().strip()
            if token and token.startswith("sk-") and len(token) >= 32:
                ADMIN_TOKEN = token
                logger.info(f"Loaded admin token from {token_file}")
                return
    token = f"sk-{secrets.token_hex(16)}"
    with open(token_file, "w") as f:
        f.write(token)
    ADMIN_TOKEN = token
    logger.info(f"Generated new admin token and saved to {token_file}: {token}")


def get_multimodal_headers():
    global ADMIN_TOKEN
    headers = {"Content-Type": "application/json"}
    MULTIMODAL_ADMIN_TOKEN = os.getenv("MULTIMODAL_ADMIN_TOKEN", ADMIN_TOKEN)
    if MULTIMODAL_ADMIN_TOKEN is None:
        raise ValueError("MULTIMODAL_ADMIN_TOKEN is not set")
    headers["Authorization"] = f"Bearer {MULTIMODAL_ADMIN_TOKEN}"
    return headers


ensure_admin_token()
