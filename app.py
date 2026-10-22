"""Flask backend for the Grand Hotel reservation assistant.

Run:   python app.py        then open http://localhost:5000
Env:   SECRET_KEY   session-cookie signing key (set a long random value in production)
       ADMIN_TOKEN  enables GET /api/bookings (send header  X-Admin-Token: <token>)
       PORT         default 5000
       DB_PATH      default ./reservations.db
"""
import os
import uuid

from flask import Flask, jsonify, request, send_from_directory, session

import bot
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True)


def session_id():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()[:500]
    if not message:
        return jsonify(error="Please type a message."), 400
    return jsonify(reply=bot.handle(session_id(), message))


@app.post("/api/reset")
def reset():
    bot.reset(session_id())
    return jsonify(ok=True)


@app.get("/api/bookings")
def bookings():
    token = os.environ.get("ADMIN_TOKEN")
    if not token or request.headers.get("X-Admin-Token") != token:
        return jsonify(error="Forbidden"), 403
    return jsonify(bookings=db.list_upcoming(bot.today().isoformat()))


# ---------------- POS endpoints ----------------
@app.get("/api/pos/items")
def pos_items():
    return jsonify(items=db.pos_items())


@app.post("/api/pos/charge")
def pos_charge():
    data = request.get_json(silent=True) or {}
    guest = str(data.get("guest", "")).strip()
    item = str(data.get("item", "")).strip()
    qty = int(data.get("qty", 1) or 1)
    if not guest or not item:
        return jsonify(error="guest and item required"), 400
    items = {it["name"]: it for it in db.pos_items()}
    if item not in items:
        return jsonify(error="unknown item"), 400
    it = items[item]
    charge = db.add_charge(guest, item, it["price_kes"], qty)
    return jsonify(charge=charge, folio=db.folio_details(guest, bot.today().isoformat()))


@app.get("/api/pos/folio/<guest>")
def pos_folio(guest):
    return jsonify(folio=db.folio_details(guest, bot.today().isoformat()))


@app.post("/api/pos/pay")
def pos_pay():
    data = request.get_json(silent=True) or {}
    guest = str(data.get("guest", "")).strip()
    amount = int(data.get("amount", 0) or 0)
    method = str(data.get("method", "")).strip()
    if not guest or not method:
        return jsonify(error="guest and method required"), 400
    out = db.outstanding(guest, bot.today().isoformat())
    if amount < 1:
        amount = out
    if amount > out:
        return jsonify(error=f"amount exceeds outstanding {out}"), 400
    payment = db.record_payment(guest, amount, method)
    details = db.folio_details(guest, bot.today().isoformat())
    return jsonify(payment=payment, receipt=bot.receipt_text(guest, details, payment),
                   outstanding=details["outstanding"])


@app.get("/api/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
