import secrets
import redis

class TokenService:
    def __init__(self, expiry=3600, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.expiry = expiry

    def generate_token(self, user_id) -> str:
        old_token = self.redis_client.get(f"user:{user_id}")
        if old_token:
            self.redis_client.delete(old_token)

        token = str(secrets.token_urlsafe(32))
        self.redis_client.set(token, user_id, ex=self.expiry)
        self.redis_client.set(f"user:{user_id}", token, ex=self.expiry)
        return token

    def invalidate_token(self, token: str):
        user_id = self.redis_client.get(token)
        if user_id:
            self.redis_client.delete(f"user:{user_id}")
        self.redis_client.delete(token)

    def validate_token(self, token: str):
        user_id = self.redis_client.get(token)
        if user_id:
            self.redis_client.expire(token, self.expiry)
        return user_id if user_id else None