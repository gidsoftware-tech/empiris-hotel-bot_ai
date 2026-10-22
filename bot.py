"""Grand Hotel - room reservation AI assistant (conversation engine).

How it is put together:
- Intent model : TF-IDF + Logistic Regression (scikit-learn)
- FAQs         : answered by the same intent model, from the FAQ table below
- Slot filling : regex extraction (room type, check-in, nights, guests, name)
- Storage      : SQLite via db.py, with atomic per-room-type availability

Used by the Flask website (app.py) and by the console (python bot.py).
"""
import re
import threading
from datetime import datetime, timedelta, timezone

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import db

# ---------------- 0. Clock (Nairobi time, UTC+3, no DST) ----------------
LOCAL_TZ = timezone(timedelta(hours=3))


def now():
    """Current local time as a naive datetime, whatever the server's clock."""
    return datetime.now(LOCAL_TZ).replace(tzinfo=None)


def today():
    return now().date()


# ---------------- 1. FAQs (edit the answers to match your hotel) ----------------
FAQ = {
    "faq_checkin": (
        ["what time is check in", "what are your check in and check out times",
         "when can i check in", "check in time", "what time is check out",
         "can i check in early", "is early check in possible", "late check out"],
        "Check-in is from 3:00 PM and check-out is by 11:00 AM. Early check-in and late "
        "check-out are subject to availability and may carry a fee."),
    "faq_cancellation": (
        ["do you offer free cancellation", "can i cancel for free", "what is your "
         "cancellation policy", "cancellation fee"],
        "Free cancellation is available up to 24 hours before your arrival date. "
        "Cancellations within 24 hours incur a one-night charge."),
    "faq_breakfast": (
        ["is breakfast included", "does the rate include breakfast", "do you serve breakfast",
         "breakfast price"],
        "Breakfast is included in our Bed & Breakfast rate plans. For other rate plans it "
        "can be added for KES 1,500 per person per day."),
    "faq_pets": (
        ["do you allow pets", "can i bring my pet", "what is the pet policy",
         "are dogs allowed", "pet fee"],
        "We welcome pets under 25 kg in designated pet-friendly rooms for a fee of KES 3,000 "
        "per night. Guide dogs stay free."),
    "faq_wifi": (
        ["do you have wifi", "is there free internet", "wifi password", "do you offer free wi-fi"],
        "Yes, complimentary high-speed Wi-Fi is available throughout the hotel. The network "
        "name and password are printed on your key card."),
    "faq_pool": (
        ["is there a swimming pool", "do you have a pool", "pool opening hours"],
        "Yes, we have an outdoor pool open from 7:00 AM to 9:00 PM. Towels are available at "
        "the pool deck."),
    "faq_gym": (
        ["do you have a gym", "is there a fitness center", "gym opening hours"],
        "Yes, our fitness center is open 24 hours and includes cardio machines, free weights, "
        "and yoga mats."),
    "faq_shuttle": (
        ["do you provide airport transportation", "is there an airport shuttle",
         "do you have a shuttle service", "airport transfer"],
        "We offer a complimentary shuttle to and from the airport between 5:00 AM and "
        "11:00 PM. Please request it at least 2 hours in advance."),
    "faq_airport": (
        ["how far are you from the airport", "how long to the airport"],
        "We are a 15-minute drive from the local airport. Our complimentary shuttle runs "
        "every 30 minutes."),
    "faq_parking": (
        ["is parking available", "do you have parking", "how much is parking",
         "is there parking and does it cost extra", "car park"],
        "Yes, valet parking is available for KES 2,000 per night and self-parking for KES "
        "1,500 per night. We also have 4 EV charging stations."),
    "faq_restaurant": (
        ["is there a restaurant", "do you serve food", "what are restaurant hours",
         "is there room service"],
        "Yes, Horizon Restaurant is open for breakfast (6:30-10:30), lunch (12:00-14:30), and "
        "dinner (18:00-22:30). Room service is available 24 hours."),
    "faq_rates": (
        ["what are your room rates", "how much is a room", "what is the room rate for a "
         "standard room", "what room types do you have", "how much per night"],
        "Standard rooms start at KES 12,000, Deluxe at KES 18,000, Family at KES 25,000 and "
        "Suites at KES 35,000 per night. Prices vary by season."),
    "faq_family": (
        ["do you have family rooms", "rooms for children", "how many people can a family room "
         "sleep"],
        "Yes, our Family Deluxe room sleeps up to 5 and includes a king bed plus a bunk bed. "
        "Free rollaway beds are available."),
    "faq_accessibility": (
        ["are there rooms for guests with disabilities", "do you have accessible rooms",
         "wheelchair accessible room"],
        "Yes, we have fully accessible rooms with roll-in showers, grab bars, and lowered "
        "amenities. Please mention the request when booking."),
    "faq_minibar": (
        ["do the rooms have minibars", "is there a safe in the room", "do rooms have a coffee "
         "maker"],
        "All rooms include a complimentary minibar (water and soft drinks), an in-room safe, "
        "and a coffee maker."),
    "faq_deposit": (
        ["how much is the deposit", "is a deposit required", "do you charge a deposit",
         "incidental deposit"],
        "A KES 10,000 incidental deposit is required at check-in. It is refunded at check-out "
        "unless there are charges."),
    "faq_smoking": (
        ["is smoking allowed", "can i smoke in the room", "smoking policy"],
        "No, all rooms are non-smoking. We have designated smoking areas outside. A cleaning "
        "fee of KES 25,000 applies for smoking in rooms."),
    "faq_currency": (
        ["what currencies do you accept", "do you accept dollars", "which payment methods do "
         "you accept", "can i pay by card"],
        "We accept major credit cards, debit cards, and cash. Foreign currency can be "
        "converted at the front desk."),
    "faq_conference": (
        ["do you have conference rooms", "do you have meeting rooms",
         "can i host an event", "wedding venue"],
        "Yes, we have 4 meeting rooms with a combined capacity of 120 people, plus AV "
        "equipment and catering."),
    "faq_spa": (
        ["is there a spa", "do you offer massages", "spa opening hours"],
        "Yes, our Serenity Spa offers massages, facials, and sauna. Open 10:00 AM to 8:00 PM. "
        "Book at the front desk."),
    "faq_frontdesk": (
        ["what time is the front desk open", "is reception open at night",
         "is there a 24 hour front desk"],
        "Our front desk is open 24 hours a day, 7 days a week."),
    "faq_walkin": (
        ["do you accept walk in guests", "can i just show up", "are walk ins welcome"],
        "Yes, walk-in guests are welcome, subject to room availability. We recommend calling "
        "ahead during peak season."),
    "faq_modify": (
        ["how do i modify my booking", "can i change my reservation",
         "how do i change my dates"],
        "You can modify or cancel through the Manage Reservation link in your confirmation "
        "email, or by calling us directly at 1-800-555-0100."),
    "faq_contact": (
        ["what is your phone number", "how can i contact you", "what is your email address",
         "talk to a human", "speak to the manager"],
        "Call 1-800-555-0100 or email reservations@grandhotel.example."),
}

# ---------------- 2. Intent model ----------------
TRAIN = {
    "book": [
        "book a room", "i want to reserve a room", "make a reservation",
        "i'd like to book a room for two people", "can i get a room tonight",
        "book a suite for 2 nights", "i need a room for the weekend",
        "reserve a double room for tomorrow", "i would like a room please",
        "can i book accommodation", "i want to stay for three nights"],
    "cancel": [
        "cancel my reservation", "i want to cancel my booking",
        "please cancel the room", "remove my reservation", "cancel booking under mary",
        "i can not come anymore please cancel", "delete my booking", "cancel my room"],
    "check": [
        "check my reservation", "do i have a booking", "what is my reservation",
        "show my booking", "check booking for name is john",
        "do i have a reservation under mary", "find my booking", "view my reservation",
        "my booking details"],
    "greet": ["hi", "hello", "hey there", "good evening", "jambo", "habari",
              "good morning", "good afternoon"],
    "bye": ["bye", "goodbye", "thanks that's all", "see you", "quit", "exit"],
    "thanks": ["thanks", "thank you", "thank you very much", "asante", "much appreciated"],
    "ack": ["yes", "ok", "okay", "alright", "sure", "fine", "cool", "great", "yeah", "please do"],
    "other": [
        "what is the weather", "who won the match", "tell me a joke", "how is the traffic",
        "what is 2+2", "do you sell cars", "what is your name", "who are you", "asdfgh",
        "qwerty", "blah blah", "random text", "what is the capital of france",
        "play some music", "i love football", "what is the news", "how old are you",
        "sing a song", "lorem ipsum", "i want pizza", "i want a car", "give me money",
        "i am hungry", "do you like cats"],
    "pos_order": [
        "order food", "i want to order", "can i order", "add to my room",
        "charge to my room", "put it on my bill", "order for john",
        "i would like to order", "can you bring me", "room service order",
        "add this to my folio", "charge this to my account", "i want to buy",
        "can i get a beer", "order a massage", "spa treatment for mary"],
    "pos_bill": [
        "my bill", "what is my bill", "room bill", "folio", "charges",
        "room charges", "how much do i owe", "my folio", "outstanding balance",
        "what are my charges", "show my bill", "bill for john", "check my folio"],
    "pos_pay": [
        "pay", "pay now", "pay bill", "settle", "settle bill", "settle up",
        "clear balance", "pay outstanding", "i want to pay", "can i pay",
        "payment", "pay my tab", "pay with mpesa", "pay with cash", "pay with card",
        "pay my bill", "settle my bill"],
}
for _key, (_qs, _ans) in FAQ.items():
    TRAIN[_key] = list(_qs)

# a few extra phrasings and an "other" class so off-topic text is not forced into an FAQ
TRAIN["faq_rates"] += ["prices", "how expensive", "room prices", "cost per night"]
TRAIN["faq_checkin"] += ["hours", "check in time please", "what time can i arrive"]
TRAIN["faq_contact"] += ["contact", "phone number please", "how do i call you"]
TRAIN["faq_breakfast"] += ["is breakfast free", "do you do breakfast"]
TRAIN["faq_parking"] += ["parking", "where can i park"]

_X = [p for _phrases in TRAIN.values() for p in _phrases]
_y = [_intent for _intent, _phrases in TRAIN.items() for _ in _phrases]
intent_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                             LogisticRegression(max_iter=2000, C=20))
intent_model.fit(_X, _y)

ACTION_MIN = 0.30        # min confidence to start a book / cancel / check
FAQ_MIN = 0.20           # min confidence to answer an FAQ
FAQ_MIDFLOW_MIN = 0.50   # min confidence to answer an FAQ in the middle of a booking


def classify(text):
    proba = intent_model.predict_proba([text])[0]
    i = proba.argmax()
    return intent_model.classes_[i], float(proba[i])


# ---------------- 3. Business rules ----------------
ROOMS = db.ROOMS
ROOM_ALIASES = {
    "standard": ["standard", "basic", "regular", "single", "cheapest", "cheap", "budget"],
    "deluxe": ["deluxe", "executive", "superior"],
    "family": ["family", "connecting"],
    "suite": ["suite", "presidential"],
}
HORIZON_DAYS = 365
MAX_NIGHTS = db.MAX_NIGHTS


# ---------------- 4. Slot extraction ----------------
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_RE = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
            r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NOT_NAMES = {"at", "for", "on", "tomorrow", "today", "tonight", "and", "please", "with",
             "room", "rooms", "book", "booking", "bookings", "reservation", "reservations",
             "reserve", "to", "the", "a", "an", "looking", "hoping", "trying", "wanting",
             "interested", "coming", "going", "planning", "thinking", "here", "new", "not",
             "so", "would", "want", "need", "like", "calling", "writing", "asking",
             "next", "this", "night", "nights", "week", "weekend", "from", "by", "in", "of",
             "is", "are", "will", "can", "my", "guest", "stay", "staying"} | set(WEEKDAYS)


def get_room(low):
    for room, aliases in ROOM_ALIASES.items():
        if any(a in low for a in aliases):
            return room
    return None


def get_guests(low, awaiting=False):
    if awaiting:
        m = re.fullmatch(r"\s*(\d+)\s*(?:guests?|people|persons|pax|adults?)?\s*", low)
        if m:
            return int(m.group(1))
        w = re.fullmatch(r"\s*(" + "|".join(NUM_WORDS) + r")\s*"
                         r"(?:guests?|people|persons|pax|adults?)?\s*", low)
        if w:
            return NUM_WORDS[w.group(1)]
    m = (re.search(r"(?:for|party of|group of|table of)\s*(\d+)\b"
                   r"(?!\s*(?::\d|\.\d|[ap]m\b|(?:st|nd|rd|th|of)\b|nights?|days?|"
                   + MONTH_RE + r"\b))", low)
         or re.search(r"\b(\d+)\s*(?:guests?|people|persons|pax|adults?)\b", low))
    if m:
        return int(m.group(1))
    m = re.search(r"(?:for|party of|group of)\s+(" + "|".join(NUM_WORDS) + r")\b"
                  r"(?!\s*(?:nights?|days?))", low)
    return NUM_WORDS[m.group(1)] if m else None


def get_nights(low, awaiting=False):
    m = re.search(r"\b(\d+)\s*(?:nights?|days?)\b", low)
    if m:
        return int(m.group(1))
    w = re.search(r"\b(" + "|".join(NUM_WORDS) + r")\s*(?:nights?|days?)\b", low)
    if w:
        return NUM_WORDS[w.group(1)]
    if awaiting:
        m = re.fullmatch(r"\s*(\d+)\s*(?:nights?|days?)?\s*", low)
        if m:
            return int(m.group(1))
        w = re.fullmatch(r"\s*(" + "|".join(NUM_WORDS) + r")\s*(?:nights?|days?)?\s*", low)
        if w:
            return NUM_WORDS[w.group(1)]
    return None


def get_date(text):
    low = text.lower()
    ref = today()
    if "today" in low or "tonight" in low:
        return ref
    if "tomorrow" in low:
        return ref + timedelta(days=1)
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", low)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()
        except ValueError:
            return None
    day = month = None
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?" + MONTH_RE + r"\b", low)
    if m:
        day, month = int(m.group(1)), MONTHS.index(m.group(2)[:3]) + 1
    else:
        m = re.search(r"\b" + MONTH_RE + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b", low)
        if m:
            day, month = int(m.group(2)), MONTHS.index(m.group(1)[:3]) + 1
        else:
            m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", low)       # day/month
            if m:
                day, month = int(m.group(1)), int(m.group(2))
    if day:
        try:
            d = datetime(ref.year, month, day).date()
            return d if d >= ref else datetime(ref.year + 1, month, day).date()
        except ValueError:
            return None
    for i, d in enumerate(WEEKDAYS):
        if d in low:
            return ref + timedelta(days=(i - ref.weekday()) % 7 or 7)
    return None


def get_name(text, awaiting=False):
    pattern = (r"(?:name is|my name's|under(?: the name)?" + (r"|i am|i'm|im|it's" if awaiting else "")
               + r")\s+([A-Za-z][A-Za-z'\- ]*)")
    m = re.search(pattern, text, re.I)
    if not m:
        return None
    words = []
    for w in m.group(1).split():
        if w.lower() in NOT_NAMES or len(words) == 3:
            break
        words.append(w)
    return " ".join(words).title() if words else None


def loose_name(text):
    """Find a name in something like 'cancel John' or 'check booking Mary'."""
    words = re.findall(r"\b[A-Z][A-Za-z'\-]{1,}\b", text)
    skip = {"My", "The", "I", "Do", "Can", "A", "An", "Please", "Booking", "Bookings",
            "Cancel", "Check", "Reservation", "Reservations", "Room", "Rooms"}
    words = [w for w in words if w not in skip]
    return " ".join(words[:3]) if words else None


# ---------------- 5. Conversation engine ----------------
SESSIONS = {}
LOCK = threading.RLock()
HELP = ("I can book a room for you, check or cancel a booking, order food/drinks/spa to your "
        "folio, check your bill and take payment, or answer questions about rates, check-in "
        "times, breakfast, parking, amenities and more.")


def new_state():
    return {"mode": None, "await": None, "last": None, "ids": [],
            "slots": {"room": None, "check_in": None, "nights": None,
                      "guests": None, "name": None},
            "pos": {"item": None, "qty": 1, "guest": None,
                    "method": None, "amount": None}}


def format_kes(amount):
    return f"KES {amount:,}"


def filled(slots):
    return sum(v is not None for v in slots.values())


def prompt_missing(s):
    for key, msg in [("room", "Which room type would you like? Standard, Deluxe, Family or Suite."),
                     ("check_in", "What date will you check in? (today, tomorrow, a weekday, "
                                  "12 October, or YYYY-MM-DD)"),
                     ("nights", "How many nights will you stay?"),
                     ("guests", "How many guests will be staying?"),
                     ("name", "What name should I put the booking under?")]:
        if s["slots"][key] is None:
            s["await"] = key
            return msg
    return None


def faq_answer(text, minimum=FAQ_MIDFLOW_MIN):
    intent, conf = classify(text)
    return FAQ[intent][1] if intent in FAQ and conf >= minimum else None


def finalize(s):
    sl = s["slots"]
    room, check_in, nights, guests, name = (sl["room"], sl["check_in"], sl["nights"],
                                            sl["guests"], sl["name"])
    s.update(new_state())
    ref = today()
    if check_in < ref:
        return "That check-in date is in the past. Please start a new booking."
    if (check_in - ref).days > HORIZON_DAYS:
        return f"I can only take bookings up to {HORIZON_DAYS} days ahead. Please start a new booking."
    if nights < 1 or nights > MAX_NIGHTS:
        return (f"Stays are between 1 and {MAX_NIGHTS} nights. "
                "Please start a new booking with a shorter stay.")
    if guests < 1:
        return "The number of guests must be at least 1. Please start a new booking."
    r = ROOMS[room]
    if guests > r["sleeps"]:
        return (f"A {room} room sleeps up to {r['sleeps']} guests. "
                "Please start a new booking and choose a larger room type.")
    check_out = check_in + timedelta(days=nights)
    booking = db.book_room(name, room, guests, check_in, check_out)
    if not booking:
        return (f"Sorry, no {room} rooms are free from {check_in:%d %b} to {check_out:%d %b}. "
                "Please try other dates or another room type.")
    return (f"Booking confirmed! #{booking['id']}\n"
            f"{name}, {guests} guest(s)\n"
            f"{room.title()} room, {check_in:%a %d %b} to {check_out:%a %d %b} "
            f"({nights} night(s))\n"
            f"KES {booking['rate']:,}/night, total KES {booking['total']:,}\n"
            "We look forward to welcoming you to Empiris Guest House!")


def describe(rows):
    return "\n".join(
        f"#{r['id']}: {r['name']}, {r['room']} room, {r['guests']} guest(s), "
        f"{r['check_in']} to {r['check_out']}" for r in rows)


def lookup(s, name, cancel):
    """Show or cancel upcoming bookings. If several match a cancel, ask which one."""
    rows = db.find_bookings(name, today().isoformat())
    s.update(new_state())
    if not rows:
        return "I couldn't find an upcoming booking under that name."
    if not cancel:
        return describe(rows)
    if len(rows) == 1:
        db.delete_bookings([rows[0]["id"]])
        return "Cancelled:\n" + describe(rows)
    s["mode"], s["ids"] = "cancel_pick", [r["id"] for r in rows]
    return ("I found several bookings:\n" + describe(rows) +
            f"\nWhich one should I cancel? Reply with the booking number (e.g. {rows[0]['id']}) "
            "or 'all'.")


def lookup_id(s, booking_id, cancel):
    b = db.get_booking(booking_id)
    s.update(new_state())
    if not b:
        return f"I couldn't find a booking with number #{booking_id}."
    if cancel:
        db.delete_bookings([booking_id])
        return "Cancelled:\n" + describe([b])
    return describe([b])


def pick_booking(s, low):
    ids = s["ids"]
    if re.search(r"\ball\b", low):
        chosen = ids
    else:
        m = re.search(r"\d+", low)
        chosen = [int(m.group())] if m and int(m.group()) in ids else []
    if not chosen:
        return ("Please reply with one of these booking numbers: "
                + ", ".join(f"#{i}" for i in ids) + " (or 'all').")
    db.delete_bookings(chosen)
    s.update(new_state())
    return "Cancelled booking " + ", ".join(f"#{i}" for i in chosen) + "."


def _booking_id(text):
    m = re.search(r"(?:#|\bbooking\b|\breservation\b|\bnumber\b|\bno\b)\s*(\d+)\b", text, re.I)
    return int(m.group(1)) if m else None


# ---------------- POS helpers ----------------
def find_pos_item(text):
    """Return a pos item dict whose name appears in text, or None."""
    low = text.lower()
    best = None
    for it in db.pos_items():
        if it["name"].lower() in low:
            if best is None or len(it["name"]) > len(best["name"]):
                best = it
    return best


def pos_menu_text():
    lines = []
    by_cat = {}
    for it in db.pos_items():
        by_cat.setdefault(it["category"], []).append(it)
    for cat, items in by_cat.items():
        lines.append(f"{cat.title()}:")
        lines.extend(f"  • {it['name']} — {format_kes(it['price_kes'])}" for it in items)
    return "\n".join(lines)


def folio_text(guest_name, details):
    lines = [f"Folio for {guest_name} ({today():%d %b %Y})", ""]
    if details["charges"]:
        lines.append("Charges:")
        for c in details["charges"]:
            lines.append(f"  {c['item_name']} x{c['qty']}  {format_kes(c['total_kes'])}")
    else:
        lines.append("No charges today.")
    lines.append("")
    lines.append(f"Charges total:   {format_kes(details['folio_total'])}")
    lines.append(f"Paid:            {format_kes(details['paid_total'])}")
    lines.append(f"Outstanding:     {format_kes(details['outstanding'])}")
    return "\n".join(lines)


def receipt_text(guest_name, details, payment):
    lines = []
    lines.append("=" * 36)
    lines.append("     EMPIRIS GUEST HOUSE")
    lines.append("     Narok County, Kenya")
    lines.append(f"     Receipt: {payment['receipt_no']}")
    lines.append(f"     Date: {today():%d %b %Y}")
    lines.append("-" * 36)
    lines.append(f"Guest: {guest_name}")
    lines.append("-" * 36)
    if details["charges"]:
        lines.append("Charges:")
        for c in details["charges"]:
            lines.append(f"  {c['item_name']:<20} x{c['qty']}  {format_kes(c['total_kes'])}")
    else:
        lines.append("No new charges.")
    lines.append("-" * 36)
    lines.append(f"Subtotal:        {format_kes(details['folio_total'])}")
    lines.append(f"Paid ({payment['method']}): {format_kes(payment['amount_kes'])}")
    change = payment["amount_kes"] - details["outstanding"]
    if change > 0:
        lines.append(f"Change:          {format_kes(change)}")
    lines.append("-" * 36)
    lines.append("     Thank you for staying with us!")
    lines.append("=" * 36)
    return "\n".join(lines)


def _handle(sid, text):
    if len(SESSIONS) > 2000:                              # keep memory bounded
        SESSIONS.pop(next(iter(SESSIONS)))
    s = SESSIONS.setdefault(sid, new_state())
    text = (text or "").strip()
    low = text.lower()
    if not text:
        return "Please type a message."
    if s["mode"] and (re.search(r"\b(stop|never ?mind|abort)\b", low) or low == "cancel"):
        s.update(new_state())
        return "No problem, I've stopped that. " + HELP

    if s["mode"] == "cancel_pick":                       # several bookings matched, choose one
        return pick_booking(s, low)

    if s["mode"] == "pos_order_item":                    # waiting for the item name
        it = find_pos_item(text)
        if it:
            s["pos"]["item"] = it
            s["mode"] = "pos_order_guest"
            s["await"] = "pos_guest"
            return (f"{it['name']} — {format_kes(it['price_kes'])}. "
                    "Which guest should I charge this to?")
        return ("Sorry, I didn't recognise that item. " + pos_menu_text() +
                "\n\nWhat would you like to order?")

    if s["mode"] == "pos_order_guest":                   # waiting for the guest name
        name = get_name(text, True) or loose_name(text) or text.strip(" .!?").title()[:30]
        it = s["pos"]["item"]
        db.add_charge(name, it["name"], it["price_kes"], s["pos"]["qty"])
        s.update(new_state())
        return (f"Added {it['name']} x{s['pos']['qty']} ({format_kes(it['price_kes'])}) "
                f"to {name}'s folio.")

    if s["mode"] == "pos_bill_lookup":                   # waiting for guest name to show folio
        name = get_name(text, True) or loose_name(text)
        if not name:
            return "What name is the folio under?"
        details = db.folio_details(name, today().isoformat())
        s.update(new_state())
        s["pos"]["guest"] = name
        msg = folio_text(name, details)
        if details["outstanding"] > 0:
            msg += (f"\n\nWould you like to pay the outstanding {format_kes(details['outstanding'])} now? "
                    "Say 'pay' and choose Cash, M-Pesa or Card.")
        return msg

    if s["mode"] == "pos_pay_method":                    # waiting for payment method
        m = text.lower()
        if "m-pesa" in m or "mpesa" in m:
            method = "M-Pesa"
        elif "card" in m or "visa" in m or "mastercard" in m:
            method = "Card"
        elif "cash" in m:
            method = "Cash"
        else:
            return "Please choose Cash, M-Pesa or Card."
        s["pos"]["method"] = method
        s["mode"] = "pos_pay_amount"
        s["await"] = "pos_amount"
        out = db.outstanding(s["pos"]["guest"], today().isoformat())
        return (f"Outstanding: {format_kes(out)}. How much would you like to pay? "
                f"(Press Enter for full amount, or type an amount in KES.)")

    if s["mode"] == "pos_pay_amount":                    # waiting for payment amount
        m = re.search(r"\d[\d,]*", text.replace(",", ""))
        out = db.outstanding(s["pos"]["guest"], today().isoformat())
        amount = int(m.group()) if m else out
        if amount < 1 or amount > out:
            return f"Please enter an amount between 1 and {format_kes(out)}."
        payment = db.record_payment(s["pos"]["guest"], amount, s["pos"]["method"])
        guest = s["pos"]["guest"]
        details = db.folio_details(guest, today().isoformat())
        s.update(new_state())
        return receipt_text(guest, details, payment) + \
            f"\n\nRemaining outstanding: {format_kes(details['outstanding'])}."

    if s["mode"] in ("cancel", "check"):                 # waiting for the booking name
        name = get_name(text, True) or loose_name(text)
        if name:
            return lookup(s, name, s["mode"] == "cancel")
        bid = _booking_id(text)
        if bid is not None:
            return lookup_id(s, bid, s["mode"] == "cancel")
        return "What name is the booking under?"

    if s["mode"] == "book":                              # filling booking details
        sl, awaiting = s["slots"], s["await"]
        if awaiting == "name" and not get_name(text, True):   # a question instead of a name?
            faq = faq_answer(text)
            if faq:
                return faq + "\n\n" + prompt_missing(s)
        before = filled(sl)
        if sl["room"] is None:
            sl["room"] = get_room(low)
        if sl["check_in"] is None:
            sl["check_in"] = get_date(text)
        if sl["nights"] is None:
            sl["nights"] = get_nights(low, awaiting == "nights")
        if sl["guests"] is None:
            sl["guests"] = get_guests(low, awaiting == "guests")
        if sl["name"] is None:
            n = get_name(text, awaiting == "name")
            if n:
                sl["name"] = n
            elif awaiting == "name" and not (get_date(text) or get_nights(low) or get_guests(low)):
                sl["name"] = text.strip(" .!?").title()[:30] or None
        if sl["guests"] is not None and sl["guests"] < 1:
            sl["guests"] = None
            s["await"] = "guests"
            return "The number of guests must be at least 1. How many guests will be staying?"
        msg = prompt_missing(s)
        if msg is None:
            return finalize(s)
        if filled(sl) == before:                         # nothing understood
            faq = faq_answer(text)
            return faq + "\n\n" + msg if faq else "Sorry, I didn't catch that. " + msg
        return msg

    intent, conf = classify(text)                        # a new request
    if (intent in ("book", "cancel", "check", "pos_order", "pos_bill") and conf < ACTION_MIN) or \
            (intent in FAQ and conf < FAQ_MIN):
        intent = "other"
    prev, s["last"] = s["last"], intent

    if intent == "ack":
        if prev == "greet":                              # "yes" right after a greeting -> book
            intent = "book"
        else:
            return "Okay! " + HELP

    if intent == "book":
        s["mode"] = "book"
        sl = s["slots"]
        sl["room"] = get_room(low)
        sl["check_in"] = get_date(text)
        sl["nights"] = get_nights(low)
        sl["guests"] = get_guests(low)
        sl["name"] = get_name(text)
        if sl["guests"] is not None and sl["guests"] < 1:
            sl["guests"] = None
        return prompt_missing(s) or finalize(s)

    if intent in ("cancel", "check"):
        name = get_name(text, True) or loose_name(text)
        if name:
            return lookup(s, name, intent == "cancel")
        bid = _booking_id(text)
        if bid is not None:
            return lookup_id(s, bid, intent == "cancel")
        s["mode"] = intent
        return "Sure - what name is the booking under?"

    if intent in FAQ:
        return FAQ[intent][1]

    if intent == "greet":
        return "Hello and welcome to Empiris Guest House! " + HELP + " Would you like to make a booking?"
    if intent == "thanks":
        return "You're welcome! Anything else I can help with?"
    if intent == "bye":
        return "Goodbye - we hope to see you at Empiris Guest House soon!"
    if intent == "pos_order":
        s["mode"] = "pos_order_item"
        return ("What would you like to order?\n" + pos_menu_text() +
                "\n\nJust tell me the item name (e.g. 'Tusker Beer' or 'Massage').")
    if intent == "pos_bill":
        s["mode"] = "pos_bill_lookup"
        return "What name is the folio under?"
    if intent == "pos_pay":
        if not s["pos"].get("guest"):
            s["mode"] = "pos_bill_lookup"
            return "What name is the folio under?"
        out = db.outstanding(s["pos"]["guest"], today().isoformat())
        if out <= 0:
            s.update(new_state())
            return f"No outstanding balance for {s['pos']['guest']}."
        s["mode"] = "pos_pay_method"
        return (f"Outstanding for {s['pos']['guest']}: {format_kes(out)}. "
                "How would you like to pay — Cash, M-Pesa or Card?")
    return "I'm the Empiris Guest House assistant. " + HELP


def handle(sid, text):
    with LOCK:
        return _handle(sid, text)


def reset(sid):
    with LOCK:
        SESSIONS.pop(sid, None)


# ---------------- 6. Console version ----------------
def _console():
    print("Grand Hotel assistant. Type 'quit' to exit.\n")
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        print("bot>", handle("console", line))
        if re.search(r"\b(quit|exit|bye|goodbye|see you)\b", line.lower()):
            break


if __name__ == "__main__":
    _console()
