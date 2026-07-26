import secrets
import time

class TokenService:
    def __init__(self, expiry=3600):
        self.expiry = expiry
        self._tokens = {}   # token -> (pairing_secret, expires_at)
        self._devices = {}  # pairing_secret -> (token, expires_at)

    def _now(self):
        return time.time()

    def generate_token(self, pairing_secret: str) -> str:
        old_token, _ = self._devices.get(pairing_secret, (None, None))
        if old_token:
            self._tokens.pop(old_token, None)

        token = secrets.token_urlsafe(32)
        expires_at = self._now() + self.expiry
        self._tokens[token] = (pairing_secret, expires_at)
        self._devices[pairing_secret] = (token, expires_at)
        return token

    def invalidate_token(self, token: str):
        entry = self._tokens.pop(token, None)
        if entry:
            pairing_secret, _ = entry
            self._devices.pop(pairing_secret, None)

    def validate_token(self, token: str) -> bool:
        entry = self._tokens.get(token)
        if not entry:
            return False

        pairing_secret, expires_at = entry
        if self._now() > expires_at:
            self._tokens.pop(token, None)
            self._devices.pop(pairing_secret, None)
            return False

        new_expiry = self._now() + self.expiry
        self._tokens[token] = (pairing_secret, new_expiry)
        self._devices[pairing_secret] = (token, new_expiry)
        return True