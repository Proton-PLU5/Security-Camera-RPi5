import secrets
import redis

class TokenService:
    def __init__(self, expiry=3600, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.expiry = expiry

    def generate_token(self, pairing_secret: str) -> str:
        old_token = self.redis_client.get(f"device:{pairing_secret}")
        if old_token:
            self.redis_client.delete(old_token)

        token = str(secrets.token_urlsafe(32))
        self.redis_client.set(token, pairing_secret, ex=self.expiry)
        self.redis_client.set(f"device:{pairing_secret}", token, ex=self.expiry)
        return token

    def invalidate_token(self, token: str):
        pairing_secret = self.redis_client.get(token)
        if pairing_secret:
            self.redis_client.delete(f"device:{pairing_secret}")
        self.redis_client.delete(token)

    def validate_token(self, token: str) -> bool:
        pairing_secret = self.redis_client.get(token)
        
        # If the token is valid, refresh its expiry time
        if pairing_secret:
            self.redis_client.expire(token, self.expiry)
            self.redis_client.expire(f"device:{pairing_secret}", self.expiry)
            return True

        return False
        