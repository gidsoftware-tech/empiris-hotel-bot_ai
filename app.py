"""Flask backend for the Empiris Guest House reservation assistant.

Public chat API + staff dashboard (login, bookings, POS folio).
"""
import os
import uuid
from functools import wraps

from flask import (
    Flask, jsonify, request, send_from_directory, session,
    render_template, redirect, url_for, flash, abort
)

import bot
import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True)


# ---------------- Auth helpers ----------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


def session_id():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


# ---------------- Public pages ----------------
@app.get("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.get("/login")
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.post("/login")
def do_login():
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    user = db.verify_user(username, password)
    if not user:
        flash("Invalid username or password.", "error")
        return redirect(url_for("login"))
    session.clear()
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    session["sid"] = uuid.uuid4().hex
    next_url = request.form.get("next") or request.args.get("next") or url_for("dashboard")
    flash(f"Welcome back, {user['username']}!", "success")
    return redirect(next_url)


@app.post("/logout")
def logout():
    session.clear()
    flash("Signed out successfully.", "success")
    return redirect(url_for("login"))


# ---------------- Chat widget (public) ----------------
@app.get("/chat")
def chat_widget():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/chat")
def chat_api():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()[:500]
    if not message:
        return jsonify(error="Please type a message."), 400
    return jsonify(reply=bot.handle(session_id(), message))


@app.post("/api/reset")
def reset_api():
    bot.reset(session_id())
    return jsonify(ok=True)


# ---------------- Staff dashboard ----------------
@app.get("/dashboard")
@login_required
def dashboard():
    since = bot.today().isoformat()
    stats = db.stats_summary(since)
    bookings = db.list_upcoming(since, limit=10)
    guests = db.distinct_guests_with_charges(since)
    outstanding_folios = []
    for g in guests:
        f = db.folio_details(g, since)
        if f["outstanding"] > 0:
            outstanding_folios.append({"guest_name": g, **f})
    outstanding_folios.sort(key=lambda x: x["outstanding"], reverse=True)
    return render_template("dashboard.html",
                           stats=stats,
                           recent_bookings=bookings,
                           outstanding_folios=outstanding_folios)


@app.get("/dashboard/bookings")
@login_required
def bookings():
    since = bot.today().isoformat()
    all_bookings = db.list_upcoming(since, limit=200)
    return render_template("bookings.html", bookings=all_bookings)


@app.post("/dashboard/bookings/<int:booking_id>/cancel")
@login_required
def cancel_booking(booking_id):
    b = db.get_booking(booking_id)
    if not b:
        flash("Booking not found.", "error")
        return redirect(url_for("bookings"))
    db.delete_bookings([booking_id])
    flash(f"Booking #{booking_id} for {b['name']} cancelled.", "success")
    return redirect(url_for("bookings"))


# ---------------- POS routes ----------------
@app.get("/dashboard/pos")
@login_required
def pos():
    items = db.pos_items()
    return render_template("pos.html", items=items)


@app.post("/dashboard/pos/charge")
@login_required
def add_charge():
    guest = (request.form.get("guest_name") or "").strip()
    item = (request.form.get("item_name") or "").strip()
    qty = int(request.form.get("qty") or 1)
    if not guest or not item:
        flash("Guest name and item are required.", "error")
        return redirect(url_for("pos"))
    items = {it["name"]: it for it in db.pos_items()}
    if item not in items:
        flash("Unknown item.", "error")
        return redirect(url_for("pos"))
    it = items[item]
    db.add_charge(guest, item, it["price_kes"], qty)
    flash(f"Added {item} x{qty} (KES {it['price_kes']:,}) to {guest}'s folio.", "success")
    return redirect(url_for("folio", guest=guest))


@app.get("/dashboard/pos/folio/<guest>")
@login_required
def folio(guest):
    since = bot.today().isoformat()
    f = db.folio_details(guest, since)
    return render_template("folio.html", guest=guest, folio=f)


@app.post("/dashboard/pos/pay")
@login_required
def record_payment():
    guest = (request.form.get("guest_name") or "").strip()
    amount = int(request.form.get("amount") or 0)
    method = (request.form.get("method") or "").strip()
    if not guest or not method:
        flash("Guest and payment method are required.", "error")
        return redirect(url_for("pos"))
    out = db.outstanding(guest, bot.today().isoformat())
    if amount < 1:
        amount = out
    if amount > out:
        flash(f"Amount exceeds outstanding balance of KES {out}.", "error")
        return redirect(url_for("folio", guest=guest))
    payment = db.record_payment(guest, amount, method)
    flash(f"Payment of KES {amount:,} recorded via {method}. Receipt {payment['receipt_no']}.", "success")
    return redirect(url_for("folio", guest=guest))


# ---------------- POS API (JSON) ----------------
@app.get("/api/pos/items")
def pos_items_api():
    return jsonify(items=db.pos_items())


@app.post("/api/pos/charge")
def pos_charge_api():
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
def pos_folio_api(guest):
    return jsonify(folio=db.folio_details(guest, bot.today().isoformat()))


@app.post("/api/pos/pay")
def pos_pay_api():
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
    details = db.folio_details(guest, db.today().isoformat())
    return jsonify(payment=payment, receipt=bot.receipt_text(guest, details, payment),
                   outstanding=details["outstanding"])


# ---------------- Admin bookings (token-gated) ----------------
@app.get("/api/bookings")
def bookings_api():
    token = os.environ.get("ADMIN_TOKEN")
    if not token or request.headers.get("X-Admin-Token") != token:
        return jsonify(error="Forbidden"), 403
    return jsonify(bookings=db.list_upcoming(bot.today().isoformat()))


@app.get("/api/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
