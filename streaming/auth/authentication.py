import logging
import sqlite3
import bcrypt
import secrets
from streaming.auth.tickets import TokenService

class Authenticator():
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.token_service = TokenService()

    def generate_pairing_secret(self) -> str:
        pairing_secret = secrets.token_urlsafe(32)

        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("INSERT INTO devices (pairing_secret) VALUES (?)", (pairing_secret,))
            conn.commit()

        return pairing_secret
    
    def generate_session_token(self, pairing_secret: str) -> str:
        return self.token_service.generate_token(pairing_secret)

    def validate_session_token(self, token: str) -> bool:
        return self.token_service.validate_token(token)

    def invalidate_session_token(self, token: str):
        self.token_service.invalidate_token(token)

    def invalidate_pairing_secret(self, pairing_secret: str):
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM devices WHERE pairing_secret = ?", (pairing_secret,))
            conn.commit()
        
        self.token_service.invalidate_token(pairing_secret)