import sqlite3
import bcrypt

DB_NAME = "travel_concierge.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_calls INTEGER DEFAULT 0
        )
    ''')
    # Preferences table (Semantic Memory)
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER,
            pref_key TEXT,
            pref_value TEXT,
            PRIMARY KEY (user_id, pref_key),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        # Hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if row:
        user_id, hashed = row
        if bcrypt.checkpw(password.encode('utf-8'), hashed):
            return user_id
    return None

def get_user_quota(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT api_calls FROM users WHERE id = ?', (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return 0

def increment_user_quota(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('UPDATE users SET api_calls = api_calls + 1 WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()

def get_user_preferences(user_id):
    """Retrieve semantic memory for the user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT pref_key, pref_value FROM user_preferences WHERE user_id = ?', (user_id,))
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def update_user_preference(user_id, key, value):
    """Update semantic memory for the user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_preferences (user_id, pref_key, pref_value)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, pref_key) DO UPDATE SET pref_value = excluded.pref_value
    ''', (user_id, key, value))
    conn.commit()
    conn.close()
