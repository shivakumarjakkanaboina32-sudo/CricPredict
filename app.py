"""
CricEdge – Flask API Backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run this file to start the local server:
    python app.py

Then open your browser and go to:
    http://localhost:5000

INSTALL REQUIREMENTS FIRST:
    pip install flask pandas numpy scikit-learn
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────
print("Loading model...")

try:
    with open('model_v4.pkl', 'rb') as f:
        MODEL_DATA = pickle.load(f)
    print("✅ model_v4.pkl loaded!")
except FileNotFoundError:
    print("❌ model_v4.pkl not found — running ipl_model_v4.py first to generate it")
    MODEL_DATA = None

# ─────────────────────────────────────────
# TEAM & VENUE DATA
# ─────────────────────────────────────────
TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad',
]

TEAM_SHORT = {
    'Chennai Super Kings'        : 'CSK',
    'Delhi Capitals'             : 'DC',
    'Gujarat Titans'             : 'GT',
    'Kolkata Knight Riders'      : 'KKR',
    'Lucknow Super Giants'       : 'LSG',
    'Mumbai Indians'             : 'MI',
    'Punjab Kings'               : 'PBKS',
    'Rajasthan Royals'           : 'RR',
    'Royal Challengers Bengaluru': 'RCB',
    'Sunrisers Hyderabad'        : 'SRH',
}

TEAM_COLORS = {
    'Chennai Super Kings'        : '#F5A623',
    'Delhi Capitals'             : '#0078BC',
    'Gujarat Titans'             : '#1C1C6E',
    'Kolkata Knight Riders'      : '#3A225D',
    'Lucknow Super Giants'       : '#A72056',
    'Mumbai Indians'             : '#004BA0',
    'Punjab Kings'               : '#ED1B24',
    'Rajasthan Royals'           : '#EA1A85',
    'Royal Challengers Bengaluru': '#EC1C24',
    'Sunrisers Hyderabad'        : '#F7A721',
}

VENUES = [
    'Wankhede Stadium',
    'MA Chidambaram Stadium, Chepauk, Chennai',
    'Eden Gardens',
    'Arun Jaitley Stadium',
    'Rajiv Gandhi International Stadium, Uppal',
    'Narendra Modi Stadium, Ahmedabad',
    'M Chinnaswamy Stadium',
    'Sawai Mansingh Stadium',
    'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
    'Dr DY Patil Sports Academy',
    'Brabourne Stadium',
]

TEAM_NAME_MAP = {
    'Delhi Daredevils'            : 'Delhi Capitals',
    'Rising Pune Supergiant'      : 'Rising Pune Supergiants',
    'Deccan Chargers'             : 'Sunrisers Hyderabad',
    'Kings XI Punjab'             : 'Punjab Kings',
    'Royal Challengers Bangalore' : 'Royal Challengers Bengaluru',
}

# ─────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────
def predict_match(team1, team2, toss_winner, toss_decision='field', venue=None):
    if MODEL_DATA is None:
        return None, None, "Model not loaded"

    model            = MODEL_DATA['model']
    FC               = MODEL_DATA['feature_cols']
    stats            = MODEL_DATA['stats']
    matches          = MODEL_DATA['matches']
    twh              = MODEL_DATA['team_win_history']
    venue_avg_score  = MODEL_DATA['venue_avg_score']
    venue_avg_wkts   = MODEL_DATA['venue_avg_wkts']
    ROLLING_METRICS  = MODEL_DATA['ROLLING_METRICS']
    match_elos       = MODEL_DATA['match_elos']
    final_elo        = MODEL_DATA['final_elo_ratings']

    team1       = TEAM_NAME_MAP.get(team1, team1)
    team2       = TEAM_NAME_MAP.get(team2, team2)
    toss_winner = TEAM_NAME_MAP.get(toss_winner, toss_winner)

    next_id = matches['id'].max() + 1

    def wr(team, n=5):
        h = twh.get(team, [])
        return float(np.mean(h[-n:])) if h else 0.5

    def streak(team):
        h = twh.get(team, [])
        s = 0
        for r in reversed(h[-5:]):
            if r == 1: s += 1
            else: break
        return s

    def sf(team):
        last = matches[matches['season'] == matches['season'].max()]
        sm = last[(last['team1'] == team) | (last['team2'] == team)]
        return float((sm['winner'] == team).sum() / len(sm)) if len(sm) > 0 else 0.5

    def rolling(team):
        rows = stats[(stats['batting_team'] == team) & (stats['match_id'] < next_id)].tail(8)
        if len(rows) == 0:
            return {m: float(stats[m].mean()) for m in ROLLING_METRICS}
        return {m: float(rows[m].mean()) for m in ROLLING_METRICS}

    t1s = rolling(team1)
    t2s = rolling(team2)

    h2h = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) |
                  ((matches['team1'] == team2) & (matches['team2'] == team1))]
    h2h5 = h2h.tail(5)
    h2h_t1  = float((h2h['winner'] == team1).sum() / len(h2h))  if len(h2h)  > 0 else 0.5
    h2h5_t1 = float((h2h5['winner'] == team1).sum() / len(h2h5)) if len(h2h5) > 0 else 0.5

    pv = matches[matches['venue'] == venue] if venue else pd.DataFrame()
    t1_vwr = float((pv['winner'] == team1).sum() / (((pv['team1'] == team1) | (pv['team2'] == team1)).sum() + 1e-9)) if venue and len(pv) > 0 else 0.5
    t2_vwr = float((pv['winner'] == team2).sum() / (((pv['team1'] == team2) | (pv['team2'] == team2)).sum() + 1e-9)) if venue and len(pv) > 0 else 0.5

    r1 = float(final_elo.get(team1, 1500))
    r2 = float(final_elo.get(team2, 1500))
    if np.isnan(r1): r1 = 1500.0
    if np.isnan(r2): r2 = 1500.0
    elo_p1 = 1 / (1 + 10 ** ((r2 - r1) / 400))

    v_avg  = float(venue_avg_score.get(venue, venue_avg_score.mean())) if venue else float(venue_avg_score.mean())
    v_wkts = float(venue_avg_wkts.get(venue, venue_avg_wkts.mean()))   if venue else float(venue_avg_wkts.mean())

    feat = {
        't1_elo': r1, 't2_elo': r2, 'elo_diff': r1 - r2, 'elo_win_prob_t1': elo_p1,
        't1_form_5': wr(team1, 5),   't2_form_5': wr(team2, 5),
        't1_form_10': wr(team1, 10), 't2_form_10': wr(team2, 10),
        'form_diff': wr(team1, 5) - wr(team2, 5),
        't1_streak': streak(team1),  't2_streak': streak(team2),
        'streak_diff': streak(team1) - streak(team2),
        'h2h_ratio': h2h_t1, 'h2h_recent': h2h5_t1,
        'toss_win_t1': 1 if toss_winner == team1 else 0,
        'bat_first': 1 if toss_decision == 'bat' else 0,
        'venue_avg_score': v_avg, 'venue_avg_wkts': v_wkts,
        't1_venue_wr': t1_vwr, 't2_venue_wr': t2_vwr,
        'venue_wr_diff': t1_vwr - t2_vwr,
        't1_season_form': sf(team1), 't2_season_form': sf(team2),
        'season_form_diff': sf(team1) - sf(team2),
    }

    for m in ROLLING_METRICS:
        feat[f't1_{m}'] = t1s[m]
        feat[f't2_{m}'] = t2s[m]
        feat[f'diff_{m}'] = t1s[m] - t2s[m]

    X    = pd.DataFrame([feat])[FC].fillna(0)
    prob = model.predict_proba(X)[0]

    model_p1 = float(prob[1])
    final_p1 = 0.70 * model_p1 + 0.30 * elo_p1
    final_p2 = 1 - final_p1

    # Key factors
    factors = []
    if abs(wr(team1, 5) - wr(team2, 5)) > 0.1:
        better = team1 if wr(team1, 5) > wr(team2, 5) else team2
        factors.append(f"{TEAM_SHORT.get(better, better)} in stronger recent form")
    if h2h_t1 > 0.6:
        factors.append(f"{TEAM_SHORT.get(team1, team1)} leads H2H ({int(h2h_t1*100)}%)")
    elif h2h_t1 < 0.4:
        factors.append(f"{TEAM_SHORT.get(team2, team2)} leads H2H ({int((1-h2h_t1)*100)}%)")
    if r1 > r2 + 50:
        factors.append(f"{TEAM_SHORT.get(team1, team1)} higher ELO rating ({int(r1)} vs {int(r2)})")
    elif r2 > r1 + 50:
        factors.append(f"{TEAM_SHORT.get(team2, team2)} higher ELO rating ({int(r2)} vs {int(r1)})")
    factors.append(f"{TEAM_SHORT.get(toss_winner, toss_winner)} won toss → chose to {toss_decision}")

    return round(final_p1 * 100, 1), round(final_p2 * 100, 1), factors

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def home():
    return render_template_string(HTML_PAGE,
        teams=TEAMS,
        venues=VENUES,
        team_colors=TEAM_COLORS,
        team_short=TEAM_SHORT,
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    team1         = data.get('team1')
    team2         = data.get('team2')
    toss_winner   = data.get('toss_winner')
    toss_decision = data.get('toss_decision', 'field')
    venue         = data.get('venue') or None

    if not team1 or not team2 or not toss_winner:
        return jsonify({'error': 'Missing required fields'}), 400

    if team1 == team2:
        return jsonify({'error': 'Team 1 and Team 2 cannot be the same'}), 400

    p1, p2, factors = predict_match(team1, team2, toss_winner, toss_decision, venue)

    if p1 is None:
        return jsonify({'error': 'Model not loaded. Run ipl_model_v4.py first.'}), 500

    winner = team1 if p1 >= p2 else team2

    return jsonify({
        'team1'       : team1,
        'team2'       : team2,
        'team1_pct'   : p1,
        'team2_pct'   : p2,
        'winner'      : winner,
        'confidence'  : max(p1, p2),
        'factors'     : factors,
        'team1_short' : TEAM_SHORT.get(team1, team1),
        'team2_short' : TEAM_SHORT.get(team2, team2),
        'team1_color' : TEAM_COLORS.get(team1, '#ffffff'),
        'team2_color' : TEAM_COLORS.get(team2, '#ffffff'),
    })

@app.route('/teams')
def get_teams():
    return jsonify({'teams': TEAMS})

@app.route('/elo')
def get_elo():
    if MODEL_DATA is None:
        return jsonify({'error': 'Model not loaded'}), 500
    final_elo = MODEL_DATA['final_elo_ratings']
    rankings = []
    for team in TEAMS:
        rating = final_elo.get(team, 1500)
        if isinstance(rating, float) and np.isnan(rating):
            rating = 1500.0
        rankings.append({'team': team, 'short': TEAM_SHORT.get(team, team),
                         'rating': round(float(rating)), 'color': TEAM_COLORS.get(team, '#fff')})
    rankings.sort(key=lambda x: -x['rating'])
    for i, r in enumerate(rankings): r['rank'] = i + 1
    return jsonify({'rankings': rankings})

# ─────────────────────────────────────────
# HTML PAGE (served from Flask directly)
# ─────────────────────────────────────────
HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CricEdge – IPL Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Teko:wght@400;500;600;700&family=Mulish:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#060910;--surface:#0d1117;--card:#111827;--border:#1f2937;--text:#f0f4f8;--muted:#6b7280;--accent:#10b981;--accent2:#3b82f6;--gold:#f59e0b;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'Mulish',sans-serif;min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 80% 50% at 50% -10%,rgba(16,185,129,0.08) 0%,transparent 60%),radial-gradient(ellipse 60% 40% at 80% 100%,rgba(59,130,246,0.06) 0%,transparent 50%);pointer-events:none;z-index:0;}
.wrap{position:relative;z-index:1;max-width:900px;margin:0 auto;padding:24px 20px 60px;}
.header{text-align:center;padding:40px 0 32px;}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:99px;padding:6px 16px;margin-bottom:20px;font-size:11px;letter-spacing:3px;text-transform:uppercase;color:var(--accent);font-weight:700;}
h1{font-family:'Teko',sans-serif;font-size:clamp(52px,10vw,88px);font-weight:700;line-height:0.9;background:linear-gradient(135deg,#fff 0%,#a7f3d0 50%,var(--accent) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.sub{color:var(--muted);font-size:15px;margin-top:12px;font-weight:300;}
.card{background:var(--card);border:1px solid var(--border);border-radius:20px;overflow:hidden;margin-bottom:24px;}
.teams-row{display:grid;grid-template-columns:1fr 60px 1fr;align-items:center;padding:28px;}
.vs{text-align:center;font-family:'Teko',sans-serif;font-size:28px;color:var(--muted);}
.team-slot{display:flex;flex-direction:column;gap:4px;}
.team-slot label{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:6px;}
.team-slot.right{align-items:flex-end;}
select{background:var(--surface);border:2px solid var(--border);border-radius:12px;padding:12px 16px;color:var(--text);font-size:14px;font-family:'Mulish',sans-serif;font-weight:600;width:100%;outline:none;cursor:pointer;transition:border-color 0.2s;}
select:focus{border-color:var(--accent);}
select option{background:#161d2a;}
.settings{border-top:1px solid var(--border);padding:20px 28px;display:grid;grid-template-columns:1fr 1fr 2fr;gap:16px;}
.setting label{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-weight:700;display:block;margin-bottom:8px;}
.seg{display:flex;gap:4px;background:var(--surface);border-radius:10px;padding:4px;}
.seg-btn{flex:1;padding:8px 12px;border:none;background:transparent;color:var(--muted);border-radius:7px;cursor:pointer;font-size:13px;font-weight:700;transition:all 0.2s;font-family:'Mulish',sans-serif;}
.seg-btn.active{background:var(--card);color:var(--text);box-shadow:0 2px 8px rgba(0,0,0,0.3);}
.predict-wrap{padding:0 28px 28px;}
.predict-btn{width:100%;padding:18px;background:var(--accent);border:none;border-radius:14px;color:#000;font-family:'Teko',sans-serif;font-size:24px;font-weight:600;letter-spacing:2px;cursor:pointer;transition:all 0.2s;}
.predict-btn:hover{background:#059669;transform:translateY(-1px);}
.predict-btn:disabled{background:var(--border);color:var(--muted);cursor:not-allowed;transform:none;}
.result{display:none;margin-bottom:24px;}
.result.show{display:block;animation:fadeUp 0.4s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
.res-header{padding:24px 28px 0;display:flex;justify-content:space-between;align-items:center;}
.res-label{font-size:11px;letter-spacing:3px;text-transform:uppercase;color:var(--accent);font-weight:700;}
.conf-pill{background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:99px;padding:4px 14px;font-size:12px;color:var(--accent);font-weight:700;}
.res-teams{padding:20px 28px;display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:center;}
.res-team{display:flex;flex-direction:column;align-items:center;gap:8px;text-align:center;}
.res-logo{width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Teko',sans-serif;font-size:18px;color:#fff;font-weight:600;}
.res-name{font-weight:700;font-size:13px;line-height:1.3;}
.res-pct{font-family:'Teko',sans-serif;font-size:40px;font-weight:700;line-height:1;}
.win{color:var(--accent);}
.lose{color:var(--muted);}
.vs-div{font-family:'Teko',sans-serif;font-size:18px;color:var(--muted);}
.bars{padding:0 28px 8px;}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:6px;}
.bar-lbl{font-size:12px;font-weight:700;width:40px;}
.bar-track{flex:1;height:10px;background:var(--surface);border-radius:99px;overflow:hidden;}
.bar-fill{height:100%;border-radius:99px;width:0;transition:width 0.8s cubic-bezier(0.4,0,0.2,1);}
.bar-pct{font-size:12px;font-weight:800;width:40px;text-align:right;}
.winner-banner{margin:0 28px 20px;padding:16px 20px;border-radius:12px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);display:flex;align-items:center;gap:12px;}
.winner-icon{font-size:24px;}
.winner-title{font-family:'Teko',sans-serif;font-size:20px;letter-spacing:1px;}
.winner-sub{font-size:12px;color:var(--muted);margin-top:2px;}
.factors{padding:0 28px 24px;}
.factors-title{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:10px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;}
.chip{background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:99px;padding:6px 14px;font-size:12px;font-weight:600;color:var(--muted);}
.elo-card{margin-bottom:24px;}
.elo-title{font-family:'Teko',sans-serif;font-size:28px;letter-spacing:1px;padding:24px 28px 16px;}
.elo-list{padding:0 28px 24px;display:flex;flex-direction:column;gap:8px;}
.elo-row{display:flex;align-items:center;gap:12px;}
.elo-rank{font-family:'Teko',sans-serif;font-size:20px;color:var(--muted);width:24px;}
.elo-logo{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Teko',sans-serif;font-size:12px;color:#fff;font-weight:600;flex-shrink:0;}
.elo-name{flex:1;font-size:13px;font-weight:700;}
.elo-bar-wrap{width:120px;}
.elo-bar-track{height:6px;background:var(--surface);border-radius:99px;overflow:hidden;}
.elo-bar-fill{height:100%;border-radius:99px;}
.elo-rating{font-family:'Teko',sans-serif;font-size:18px;width:48px;text-align:right;}
.loading{text-align:center;padding:20px;color:var(--muted);font-size:14px;display:none;}
.error-msg{background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:14px 18px;color:#ef4444;font-size:13px;margin:0 28px 20px;display:none;}
.note{text-align:center;font-size:12px;color:var(--muted);margin-top:8px;}
@media(max-width:600px){.teams-row,.settings{grid-template-columns:1fr;padding:16px;}.vs{display:none;}.settings{grid-template-columns:1fr 1fr;}.predict-wrap{padding:0 16px 20px;}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="badge">🏏 Live Prediction · Flask API</div>
    <h1>CRIC<br>EDGE</h1>
    <p class="sub">IPL Match Prediction · Powered by ELO + Statistical Modelling</p>
  </div>

  <div class="card">
    <div class="teams-row">
      <div class="team-slot">
        <label>Team 1</label>
        <select id="team1" onchange="checkReady()">
          <option value="">Select Team 1</option>
          {% for team in teams %}<option value="{{ team }}">{{ team }}</option>{% endfor %}
        </select>
      </div>
      <div class="vs">VS</div>
      <div class="team-slot right">
        <label>Team 2</label>
        <select id="team2" onchange="checkReady()">
          <option value="">Select Team 2</option>
          {% for team in teams %}<option value="{{ team }}">{{ team }}</option>{% endfor %}
        </select>
      </div>
    </div>

    <div class="settings">
      <div class="setting">
        <label>Toss Winner</label>
        <div class="seg">
          <button class="seg-btn active" id="toss-t1" onclick="setToss('t1')">Team 1</button>
          <button class="seg-btn" id="toss-t2" onclick="setToss('t2')">Team 2</button>
        </div>
      </div>
      <div class="setting">
        <label>Toss Decision</label>
        <div class="seg">
          <button class="seg-btn active" id="dec-field" onclick="setDecision('field')">Field</button>
          <button class="seg-btn" id="dec-bat" onclick="setDecision('bat')">Bat</button>
        </div>
      </div>
      <div class="setting">
        <label>Venue (Optional)</label>
        <select id="venue">
          <option value="">Any Venue</option>
          {% for venue in venues %}<option value="{{ venue }}">{{ venue }}</option>{% endfor %}
        </select>
      </div>
    </div>

    <div class="predict-wrap">
      <div class="loading" id="loading">⚙️ Running prediction model...</div>
      <button class="predict-btn" id="predict-btn" onclick="runPrediction()" disabled>
        SELECT BOTH TEAMS
      </button>
    </div>
  </div>

  <!-- Result Card -->
  <div class="card result" id="result-card">
    <div class="res-header">
      <div class="res-label">🔮 Prediction Result</div>
      <div class="conf-pill" id="conf-pill"></div>
    </div>
    <div class="error-msg" id="error-msg"></div>
    <div class="res-teams">
      <div class="res-team">
        <div class="res-logo" id="res-logo1"></div>
        <div class="res-name" id="res-name1"></div>
        <div class="res-pct" id="res-pct1"></div>
      </div>
      <div class="vs-div">VS</div>
      <div class="res-team">
        <div class="res-logo" id="res-logo2"></div>
        <div class="res-name" id="res-name2"></div>
        <div class="res-pct" id="res-pct2"></div>
      </div>
    </div>
    <div class="bars">
      <div class="bar-row">
        <div class="bar-lbl" id="bar-lbl1"></div>
        <div class="bar-track"><div class="bar-fill" id="bar-fill1"></div></div>
        <div class="bar-pct" id="bar-pct1"></div>
      </div>
      <div class="bar-row">
        <div class="bar-lbl" id="bar-lbl2"></div>
        <div class="bar-track"><div class="bar-fill" id="bar-fill2"></div></div>
        <div class="bar-pct" id="bar-pct2"></div>
      </div>
    </div>
    <div class="winner-banner" id="winner-banner">
      <div class="winner-icon">🏆</div>
      <div>
        <div class="winner-title" id="winner-title"></div>
        <div class="winner-sub" id="winner-sub"></div>
      </div>
    </div>
    <div class="factors">
      <div class="factors-title">💡 Key Factors</div>
      <div class="chips" id="chips"></div>
    </div>
  </div>

  <!-- ELO Rankings -->
  <div class="card elo-card" id="elo-card">
    <div class="elo-title">⚡ ELO POWER RANKINGS</div>
    <div class="elo-list" id="elo-list">Loading...</div>
  </div>

  <div class="note">Model trained on IPL 2008–2024 · 1,090 matches · 57 features · Running locally via Flask</div>
</div>

<script>
const COLORS = {{ team_colors|tojson }};
const SHORT  = {{ team_short|tojson }};
let toss = 't1', decision = 'field';

function setToss(t) {
  toss = t;
  document.getElementById('toss-t1').classList.toggle('active', t==='t1');
  document.getElementById('toss-t2').classList.toggle('active', t==='t2');
  const t1 = document.getElementById('team1').value;
  const t2 = document.getElementById('team2').value;
  document.getElementById('toss-t1').textContent = t1 ? SHORT[t1]||'Team 1' : 'Team 1';
  document.getElementById('toss-t2').textContent = t2 ? SHORT[t2]||'Team 2' : 'Team 2';
}

function setDecision(d) {
  decision = d;
  document.getElementById('dec-field').classList.toggle('active', d==='field');
  document.getElementById('dec-bat').classList.toggle('active', d==='bat');
}

function checkReady() {
  const t1 = document.getElementById('team1').value;
  const t2 = document.getElementById('team2').value;
  const btn = document.getElementById('predict-btn');
  document.getElementById('toss-t1').textContent = t1 ? SHORT[t1]||t1 : 'Team 1';
  document.getElementById('toss-t2').textContent = t2 ? SHORT[t2]||t2 : 'Team 2';
  if (t1 && t2 && t1 !== t2) {
    btn.disabled = false;
    btn.textContent = `PREDICT: ${SHORT[t1]||t1} VS ${SHORT[t2]||t2}`;
  } else {
    btn.disabled = true;
    btn.textContent = t1 === t2 && t1 ? 'PICK DIFFERENT TEAMS' : 'SELECT BOTH TEAMS';
  }
}

async function runPrediction() {
  const team1 = document.getElementById('team1').value;
  const team2 = document.getElementById('team2').value;
  const venue = document.getElementById('venue').value;
  const toss_winner = toss === 't1' ? team1 : team2;

  document.getElementById('loading').style.display = 'block';
  document.getElementById('predict-btn').disabled = true;
  document.getElementById('result-card').classList.remove('show');

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({team1, team2, toss_winner, toss_decision: decision, venue: venue||null})
    });
    const data = await resp.json();

    if (data.error) {
      showError(data.error); return;
    }

    const p1 = data.team1_pct, p2 = data.team2_pct;
    const winner = data.winner;

    document.getElementById('res-logo1').textContent = data.team1_short;
    document.getElementById('res-logo1').style.background = data.team1_color;
    document.getElementById('res-logo2').textContent = data.team2_short;
    document.getElementById('res-logo2').style.background = data.team2_color;
    document.getElementById('res-name1').textContent = team1;
    document.getElementById('res-name2').textContent = team2;
    document.getElementById('res-pct1').textContent = p1 + '%';
    document.getElementById('res-pct1').className = 'res-pct ' + (p1 >= p2 ? 'win' : 'lose');
    document.getElementById('res-pct2').textContent = p2 + '%';
    document.getElementById('res-pct2').className = 'res-pct ' + (p2 > p1 ? 'win' : 'lose');
    document.getElementById('conf-pill').textContent = data.confidence + '% confidence';
    document.getElementById('bar-lbl1').textContent = data.team1_short;
    document.getElementById('bar-lbl2').textContent = data.team2_short;
    document.getElementById('bar-pct1').textContent = p1 + '%';
    document.getElementById('bar-pct2').textContent = p2 + '%';

    setTimeout(() => {
      document.getElementById('bar-fill1').style.width = p1 + '%';
      document.getElementById('bar-fill1').style.background = data.team1_color;
      document.getElementById('bar-fill2').style.width = p2 + '%';
      document.getElementById('bar-fill2').style.background = data.team2_color;
    }, 100);

    const wColor = COLORS[winner] || '#10b981';
    document.getElementById('winner-title').textContent = '🏆 ' + winner + ' PREDICTED TO WIN';
    document.getElementById('winner-title').style.color = wColor;
    document.getElementById('winner-sub').textContent = data.confidence + '% win probability · Toss: ' + SHORT[toss_winner] + ' chose to ' + decision;

    document.getElementById('chips').innerHTML = data.factors.map(f =>
      `<div class="chip">${f}</div>`
    ).join('');

    document.getElementById('result-card').classList.add('show');
  } catch(e) {
    showError('Connection error — is the Flask server running?');
  } finally {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('predict-btn').disabled = false;
    checkReady();
  }
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = '❌ ' + msg;
  el.style.display = 'block';
  document.getElementById('result-card').classList.add('show');
  document.getElementById('loading').style.display = 'none';
}

// Load ELO rankings on page load
async function loadElo() {
  try {
    const resp = await fetch('/elo');
    const data = await resp.json();
    const list = document.getElementById('elo-list');
    list.innerHTML = data.rankings.map(r => `
      <div class="elo-row">
        <div class="elo-rank">${r.rank}</div>
        <div class="elo-logo" style="background:${r.color}">${r.short}</div>
        <div class="elo-name">${r.team}</div>
        <div class="elo-bar-wrap">
          <div class="elo-bar-track">
            <div class="elo-bar-fill" style="width:${((r.rating-1400)/3)}%;background:${r.color}"></div>
          </div>
        </div>
        <div class="elo-rating" style="color:${r.color}">${r.rating}</div>
      </div>
    `).join('');
  } catch(e) {
    document.getElementById('elo-list').textContent = 'Could not load ELO rankings';
  }
}

loadElo();
</script>
</body>
</html>'''

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CricEdge – IPL Prediction Server")
    print("="*50)
    print("  Open your browser and go to:")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
