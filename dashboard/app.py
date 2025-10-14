"""
File: dashboard/app.py
Purpose: Minimal Flask app placeholder for your operator dashboard.
         Later, call the FastAPI service to show live charts/alerts.

Run:
    python dashboard/app.py
Visit:
    http://127.0.0.1:5000
"""

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "<h3>Smart Edge Device Dashboard</h3><p>Coming soon: live alerts & charts.</p>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
