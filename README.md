# Grand Hotel Reservation AI Bot

A room reservation chatbot built from four files:

| file | role |
|------|------|
| `db.py` | SQLite storage layer (one locked connection, atomic booking) |
| `bot.py` | conversation engine (intent model + FAQ + slot filling) |
| `app.py` | Flask backend (`/api/chat`, `/api/reset`, `/api/health`, `/api/bookings`) |
| `index.html` | chat web UI |

The FAQ engine uses a small **TF-IDF + Logistic Regression** intent model
(scikit-learn), so there is no GPU and no model download.

## Setup

```
cd C:\Users\yoga\hotel-ai-bot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Web UI:

```
python app.py
```

Then open **http://localhost:5000**.

Console version:

```
python bot.py
```

## Example prompts

- "What time is check-in?"
- "Is breakfast included?"
- "Book a deluxe room for 2 nights from tomorrow for Alex"
- "Check my booking under Alex"
- "Cancel Alex"

## Booking rules

- Room types and inventory live in `db.ROOMS` (standard / deluxe / family / suite).
- A stay must be **1 to `db.MAX_NIGHTS` (30) nights**, and check-in can be at most
  `bot.HORIZON_DAYS` (365) days ahead and never in the past.
- Guest count must fit the chosen room's capacity (`sleeps`).
- Availability is checked inside a single `BEGIN IMMEDIATE` transaction, so two
  guests can never be sold the same last room of a type.

## Timezone

All dates use **Nairobi time (UTC+3)** regardless of the server clock
(`bot.LOCAL_TZ`).

## Environment variables

| name | purpose |
|------|---------|
| `SECRET_KEY` | session-cookie signing key — set a long random value in production |
| `ADMIN_TOKEN` | enables `GET /api/bookings` (send header `X-Admin-Token: <token>`) |
| `PORT` | Flask port, default 5000 |
| `DB_PATH` | SQLite file path, default `./reservations.db` |

## Adding your own FAQs

Add an entry to the `FAQ` dictionary in `bot.py`:

```python
"faq_pool_heating": (
    ["is the pool heated", "pool temperature"],
    "Yes, our outdoor pool is heated to 28°C year round."),
```

The keys are grouped into `TRAIN` at import time, so the intent model picks up
the new phrasings automatically. Restart the server after editing.

## Known limitations

- Cancelling only needs the guest's name — there is no login or phone check.
  Storing a phone number with each booking would fix that.
- There is no per-room assignment; a booking holds one room of the chosen type.
