import logging
import sqlite3
import bcrypt
from streaming.auth.tickets import TokenService

class Authenticator():
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.token_service = TokenService()

    def authenticate(self, username: str, password: str) -> bool:
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()

            if result is None:
                bcrypt.checkpw(password.encode(), bcrypt.gensalt()) # Perform a dummy hash to mitigate timing attacks
                return False
            
            stored_password = result[0]
            return bcrypt.checkpw(password.encode(), stored_password.encode())
        
    def generate_token(self, user_id: str) -> str:
        return self.token_service.generate_token(user_id)

    def invalidate_token(self, token: str):
        self.token_service.invalidate_token(token)

    def validate_token(self, token: str):
        return self.token_service.validate_token(token)
        
    def create_user(self, username: str, password: str) -> bool:
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            try:
                hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                return True

            except sqlite3.IntegrityError:
                logging.error(f"User {username} already exists.")
                return False
            except sqlite3.Error as e:
                logging.error(f"Database error: {e}")
                return False