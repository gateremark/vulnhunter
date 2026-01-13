"""
VulnHunter - Intentionally Vulnerable Flask Application
Used as the target environment for security RL training.
"""
import sqlite3
import os
from flask import Flask, request, render_template_string

app = Flask(__name__)
DB_PATH = "/tmp/vulnhunter.db"

def init_db():
    """Initialize the vulnerable database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            password TEXT,
            email TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS secrets (
            id INTEGER PRIMARY KEY,
            flag TEXT
        )
    """)
    # Insert test data
    cursor.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', 'supersecret123', 'admin@example.com')")
    cursor.execute("INSERT OR IGNORE INTO users VALUES (2, 'user', 'password', 'user@example.com')")
    cursor.execute("INSERT OR IGNORE INTO secrets VALUES (1, 'FLAG{vulnhunter_sql_injection_found}')")
    conn.commit()
    conn.close()

# VULNERABILITY 1: SQL Injection in login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        
        # VULNERABLE: Direct string formatting in SQL query
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        try:
            cursor.execute(query)
            user = cursor.fetchone()
            if user:
                return f"Welcome {user[1]}!"
            return "Invalid credentials"
        except Exception as e:
            return f"Error: {str(e)}"
    return """
    <form method="post">
        <input name="username" placeholder="Username">
        <input name="password" type="password" placeholder="Password">
        <button type="submit">Login</button>
    </form>
    """

# VULNERABILITY 2: Reflected XSS
@app.route("/search")
def search():
    query = request.args.get("q", "")
    # VULNERABLE: No sanitization of user input in HTML
    return f"<h1>Search Results for: {query}</h1><p>No results found.</p>"

# VULNERABILITY 3: Path Traversal
@app.route("/files")
def read_file():
    filename = request.args.get("name", "readme.txt")
    # VULNERABLE: No path validation
    try:
        with open(f"/app/files/{filename}", "r") as f:
            return f"<pre>{f.read()}</pre>"
    except:
        return "File not found"

# Safe endpoint for testing
@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
