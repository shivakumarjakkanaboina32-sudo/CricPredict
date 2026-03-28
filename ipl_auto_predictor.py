"""
IPL AUTO PREDICTOR — v1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automatically fetches today's IPL match details and
runs the trained prediction model — no manual input.

SETUP (one time):
  1. Sign up free at https://cricapi.com
  2. Get your API key (free tier = 100 calls/day)
  3. Paste it below where it says YOUR_API_KEY_HERE

RUN:
  python ipl_auto_predictor.py

It will:
  • Find today's IPL match automatically
  • Extract teams, venue, toss info
  • Run the prediction model
  • Show you the result
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import requests
import json
import sys
from datetime import datetime, date

# ─────────────────────────────────────────────────────
# CONFIGURATION — paste your free API key here
# Get one free at: https://cricapi.com/register
# ─────────────────────────────────────────────────────
CRICAPI_KEY = "YOUR_API_KEY_HERE"

# ─────────────────────────────────────────────────────
# TEAM NAME NORMALISER
# Maps various API name formats → our model's names
# ─────────────────────────────────────────────────────
TEAM_NORMALISE = {
    # Full names
    "Mumbai Indians"                  : "Mumbai Indians",
    "Chennai Super Kings"             : "Chennai Super Kings",
    "Royal Challengers Bangalore"     : "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru"     : "Royal Challengers Bengaluru",
    "Kolkata Knight Riders"           : "Kolkata Knight Riders",
    "Delhi Capitals"                  : "Delhi Capitals",
    "Sunrisers Hyderabad"             : "Sunrisers Hyderabad",
    "Rajasthan Royals"                : "Rajasthan Royals",
    "Punjab Kings"                    : "Punjab Kings",
    "Kings XI Punjab"                 : "Punjab Kings",
    "Gujarat Titans"                  : "Gujarat Titans",
    "Lucknow Super Giants"            : "Lucknow Super Giants",
    # Short codes that some APIs return
    "MI"   : "Mumbai Indians",
    "CSK"  : "Chennai Super Kings",
    "RCB"  : "Royal Challengers Bengaluru",
    "KKR"  : "Kolkata Knight Riders",
    "DC"   : "Delhi Capitals",
    "SRH"  : "Sunrisers Hyderabad",
    "RR"   : "Rajasthan Royals",
    "PBKS" : "Punjab Kings",
    "GT"   : "Gujarat Titans",
    "LSG"  : "Lucknow Super Giants",
}

def normalise(name):
    """Convert any API team name format to our model's format."""
    if not name:
        return None
    # Direct match
    if name in TEAM_NORMALISE:
        return TEAM_NORMALISE[name]
    # Partial match (e.g. "Mumbai Indians Cricket Club" → "Mumbai Indians")
    for key, val in TEAM_NORMALISE.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return val
    return name  # return as-is if no match found

# ─────────────────────────────────────────────────────
# STEP 1: FETCH TODAY'S MATCHES FROM CRICAPI
# ─────────────────────────────────────────────────────

def fetch_todays_ipl_match():
    """
    Fetches today's IPL match from CricAPI.
    Returns match details dict or None.
    """
    print("🌐 Fetching today's IPL matches from CricAPI...")

    if CRICAPI_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  No API key set! Using demo mode with mock data.")
        print("   Sign up free at https://cricapi.com to get your key.\n")
        return _demo_match()

    try:
        # CricAPI endpoint for current matches
        url = f"https://api.cricapi.com/v1/currentMatches?apikey={CRICAPI_KEY}&offset=0"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if data.get('status') != 'success':
            print(f"❌ API error: {data.get('message', 'Unknown error')}")
            return None

        matches = data.get('data', [])
        today = date.today().isoformat()

        # Filter for IPL matches today
        ipl_matches = []
        for m in matches:
            name = m.get('name', '').lower()
            series = m.get('series', '').lower()
            match_date = m.get('date', '')[:10]  # YYYY-MM-DD

            is_ipl = 'ipl' in name or 'ipl' in series or \
                     'indian premier league' in series.lower()
            is_today = match_date == today

            if is_ipl:
                ipl_matches.append(m)

        if not ipl_matches:
            print(f"📅 No IPL matches found for today ({today}).")
            print("   Either it's a rest day or the season hasn't started yet.")
            print("\n💡 Switching to manual mode...")
            return _manual_input()

        # Take the first IPL match (or ask user if multiple)
        if len(ipl_matches) > 1:
            print(f"\n📋 Found {len(ipl_matches)} IPL matches today:")
            for i, m in enumerate(ipl_matches):
                print(f"   {i+1}. {m.get('name')}")
            choice = input("\nEnter match number (or 1): ").strip()
            idx = int(choice) - 1 if choice.isdigit() else 0
            match = ipl_matches[idx]
        else:
            match = ipl_matches[0]

        return _parse_match(match)

    except requests.exceptions.ConnectionError:
        print("❌ No internet connection. Switching to manual mode.")
        return _manual_input()
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return _manual_input()


def _parse_match(raw):
    """Extract relevant fields from CricAPI match object."""
    teams = raw.get('teams', [])
    toss = raw.get('tossWinner', '')
    toss_choice = raw.get('tossChoice', 'field').lower()
    venue = raw.get('venue', '')
    name = raw.get('name', '')
    match_date = raw.get('date', '')

    # Normalise team names
    team1 = normalise(teams[0]) if len(teams) > 0 else None
    team2 = normalise(teams[1]) if len(teams) > 1 else None
    toss_winner = normalise(toss) if toss else team1

    # Normalise toss decision
    if 'bat' in toss_choice:
        decision = 'bat'
    elif 'field' in toss_choice or 'bowl' in toss_choice:
        decision = 'field'
    else:
        decision = 'field'  # default

    return {
        'name'        : name,
        'team1'       : team1,
        'team2'       : team2,
        'toss_winner' : toss_winner,
        'decision'    : decision,
        'venue'       : venue,
        'date'        : match_date,
        'source'      : 'CricAPI (live)',
    }


def _demo_match():
    """Returns a realistic demo match for testing without an API key."""
    return {
        'name'        : 'Mumbai Indians vs Chennai Super Kings, IPL 2026',
        'team1'       : 'Mumbai Indians',
        'team2'       : 'Chennai Super Kings',
        'toss_winner' : 'Chennai Super Kings',
        'decision'    : 'field',
        'venue'       : 'Wankhede Stadium',
        'date'        : date.today().isoformat(),
        'source'      : 'Demo mode (no API key)',
    }


def _manual_input():
    """Fallback: ask user to enter match details manually."""
    print("\n📝 MANUAL MODE — enter today's match details:")
    print("\nAvailable teams:")
    teams = [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
        "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
        "Rajasthan Royals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"
    ]
    for i, t in enumerate(teams, 1):
        print(f"  {i:2}. {t}")

    def pick_team(prompt):
        val = input(f"\n{prompt}: ").strip()
        if val.isdigit():
            return teams[int(val)-1]
        return normalise(val) or val

    team1 = pick_team("Team 1 (name or number)")
    team2 = pick_team("Team 2 (name or number)")

    print(f"\nToss winner: 1={team1}  2={team2}")
    toss_choice = input("Toss winner (1 or 2): ").strip()
    toss_winner = team1 if toss_choice == '1' else team2

    dec = input("Toss decision (bat/field): ").strip().lower()
    decision = 'bat' if 'bat' in dec else 'field'

    venue = input("Venue (press Enter to skip): ").strip() or None

    return {
        'name'        : f"{team1} vs {team2}",
        'team1'       : team1,
        'team2'       : team2,
        'toss_winner' : toss_winner,
        'decision'    : decision,
        'venue'       : venue,
        'date'        : date.today().isoformat(),
        'source'      : 'Manual input',
    }


# ─────────────────────────────────────────────────────
# STEP 2: LOAD MODEL & PREDICT
# ─────────────────────────────────────────────────────

def load_model():
    """Load the trained IPL v3 model from pickle."""
    try:
        import pickle
        with open('model_v3.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ model_v3.pkl not found!")
        print("   Run ipl_model_v3.py first to train and save the model.")
        sys.exit(1)


def predict(model_data, team1, team2, toss_winner, decision='field', venue=None):
    """Run prediction using the trained model."""
    import pandas as pd
    import numpy as np

    model            = model_data['model']
    FEATURE_COLS     = model_data['feature_cols']
    stats            = model_data['stats']
    matches          = model_data['matches']
    team_win_history = model_data['team_win_history']
    venue_avg_score  = model_data['venue_avg_score']
    venue_avg_wkts   = model_data['venue_avg_wkts']
    ROLLING_METRICS  = model_data['ROLLING_METRICS']

    next_id = matches['id'].max() + 1

    def win_rate(team, n=10):
        h = team_win_history.get(team, [])
        return float(np.mean(h[-n:])) if h else 0.5

    def rolling(team):
        rows = stats[(stats['batting_team'] == team) &
                     (stats['match_id'] < next_id)].tail(8)
        if len(rows) == 0:
            return {m: float(stats[m].mean()) for m in ROLLING_METRICS}
        return {m: float(rows[m].mean()) for m in ROLLING_METRICS}

    t1s = rolling(team1)
    t2s = rolling(team2)

    h2h = matches[
        ((matches['team1'] == team1) & (matches['team2'] == team2)) |
        ((matches['team1'] == team2) & (matches['team2'] == team1))
    ]
    h2h_t1 = float((h2h['winner'] == team1).sum() / len(h2h)) if len(h2h) > 0 else 0.5

    v_avg  = float(venue_avg_score.get(venue, venue_avg_score.mean())) if venue else float(venue_avg_score.mean())
    v_wkts = float(venue_avg_wkts.get(venue, venue_avg_wkts.mean()))   if venue else float(venue_avg_wkts.mean())

    last = matches[matches['season'] == matches['season'].max()]
    def sform(team):
        sm = last[(last['team1'] == team) | (last['team2'] == team)]
        return float((sm['winner'] == team).sum() / len(sm)) if len(sm) > 0 else 0.5

    feat = {
        't1_form'          : win_rate(team1),
        't2_form'          : win_rate(team2),
        'form_diff'        : win_rate(team1) - win_rate(team2),
        't1_season_form'   : sform(team1),
        't2_season_form'   : sform(team2),
        'season_form_diff' : sform(team1) - sform(team2),
        'h2h_ratio'        : h2h_t1,
        'toss_win_t1'      : 1 if toss_winner == team1 else 0,
        'bat_first'        : 1 if decision == 'bat' else 0,
        'venue_avg_score'  : v_avg,
        'venue_avg_wkts'   : v_wkts,
    }
    for m in ROLLING_METRICS:
        feat[f't1_{m}']   = t1s[m]
        feat[f't2_{m}']   = t2s[m]
        feat[f'diff_{m}'] = t1s[m] - t2s[m]

    X    = pd.DataFrame([feat])[FEATURE_COLS]
    prob = model.predict_proba(X)[0]

    return float(prob[1]) * 100, float(prob[0]) * 100  # t1_pct, t2_pct


# ─────────────────────────────────────────────────────
# STEP 3: DISPLAY RESULT
# ─────────────────────────────────────────────────────

def display_result(match_info, t1_pct, t2_pct):
    team1  = match_info['team1']
    team2  = match_info['team2']
    winner = team1 if t1_pct >= t2_pct else team2
    conf   = max(t1_pct, t2_pct)

    SHORT = {
        'Mumbai Indians':'MI','Chennai Super Kings':'CSK',
        'Royal Challengers Bengaluru':'RCB','Kolkata Knight Riders':'KKR',
        'Delhi Capitals':'DC','Sunrisers Hyderabad':'SRH',
        'Rajasthan Royals':'RR','Punjab Kings':'PBKS',
        'Gujarat Titans':'GT','Lucknow Super Giants':'LSG',
    }
    s1 = SHORT.get(team1, team1[:3].upper())
    s2 = SHORT.get(team2, team2[:3].upper())

    bar1 = '█' * int(t1_pct / 2.5)
    bar2 = '█' * int(t2_pct / 2.5)

    print(f"\n{'='*62}")
    print(f"  🏏  AUTO IPL PREDICTION")
    print(f"{'='*62}")
    print(f"  Match   : {match_info['name']}")
    print(f"  Date    : {match_info['date']}")
    print(f"  Venue   : {match_info['venue'] or 'Unknown'}")
    print(f"  Toss    : {match_info['toss_winner']} won → chose to {match_info['decision']}")
    print(f"  Source  : {match_info['source']}")
    print(f"{'─'*62}")
    print(f"  {s1:<6} {bar1:<40} {t1_pct:.1f}%")
    print(f"  {s2:<6} {bar2:<40} {t2_pct:.1f}%")
    print(f"{'─'*62}")
    print(f"  🏆 PREDICTED WINNER  : {winner}")
    print(f"  📊 CONFIDENCE        : {conf:.1f}%")
    print(f"{'='*62}")

    # H2H note
    from ipl_model_v3 import matches as m_df  # reuse if available
    print(f"\n  💡 Note: This is a probabilistic prediction based on")
    print(f"     historical IPL data (2008–2024). Sport is uncertain!")
    print(f"{'─'*62}\n")

    # Save result to JSON for logging
    result = {
        'date'       : match_info['date'],
        'match'      : match_info['name'],
        'team1'      : team1,
        'team2'      : team2,
        'toss_winner': match_info['toss_winner'],
        'decision'   : match_info['decision'],
        'venue'      : match_info['venue'],
        't1_pct'     : round(t1_pct, 1),
        't2_pct'     : round(t2_pct, 1),
        'predicted_winner': winner,
        'confidence' : round(conf, 1),
        'source'     : match_info['source'],
        'timestamp'  : datetime.now().isoformat(),
    }

    # Append to prediction log
    log_file = 'prediction_log.json'
    try:
        with open(log_file) as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log = []

    log.append(result)
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"  📝 Prediction saved to {log_file}")
    return result


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*62)
    print("  IPL AUTO PREDICTOR  |  Powered by 2008–2024 ML Model")
    print("="*62)
    print(f"  Today: {date.today().strftime('%A, %d %B %Y')}\n")

    # 1. Get today's match
    match_info = fetch_todays_ipl_match()

    if not match_info or not match_info.get('team1') or not match_info.get('team2'):
        print("❌ Could not determine match teams. Exiting.")
        sys.exit(1)

    print(f"\n✅ Match found: {match_info['team1']} vs {match_info['team2']}")
    print(f"   Venue       : {match_info.get('venue', 'Unknown')}")
    print(f"   Toss        : {match_info['toss_winner']} won → {match_info['decision']}")

    # 2. Load model
    print("\n🧠 Loading prediction model...")
    model_data = load_model()
    print("✅ Model loaded!")

    # 3. Predict
    print("\n⚙️  Running prediction...")
    t1_pct, t2_pct = predict(
        model_data,
        team1        = match_info['team1'],
        team2        = match_info['team2'],
        toss_winner  = match_info['toss_winner'],
        decision     = match_info['decision'],
        venue        = match_info.get('venue'),
    )

    # 4. Display
    display_result(match_info, t1_pct, t2_pct)
