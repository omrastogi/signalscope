"""
Generate demo/signalscope_demo.html — self-contained interactive trading demo.

Usage (from project root):
    python demo/generate_demo.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rl.agent import QLearningAgent, Q_TABLE_PATH
from rl.environment import load_env, ACTIONS, STARTING_CASH, LOT_SIZE


def capture_episode(agent):
    env = load_env("test")
    agent.epsilon = 0.0
    states = env.reset()
    done = False
    days = []

    while not done:
        step_date = env.dates[env._step_idx]
        action_lots = {t: agent.act(states[t], explore=False, use_vol=True) for t in env.tickers}
        actions   = {t: al[0] for t, al in action_lots.items()}
        lot_sizes = {t: al[1] for t, al in action_lots.items()}
        next_states, ticker_rewards, _, done = env.step(actions, lot_sizes)
        days.append({
            "date": str(step_date)[:10],
            "portfolio_value": round(env.portfolio_value, 2),
            "bh_value": 0.0,  # filled later
            "actions": {t: ACTIONS[actions[t]] for t in env.tickers},
            "pnl": {t: round(ticker_rewards[t] * STARTING_CASH, 2) for t in env.tickers},
            "positions": {t: env.shares[t] for t in env.tickers},
        })
        states = next_states

    return env, days


def attach_bh_curve(env, days):
    tickers = env.tickers
    dates = env.dates
    budget = STARTING_CASH / len(tickers)
    day0 = dates[0]
    bh_shares = {
        t: budget / float(env.data_by_ticker[t].loc[day0, "close"])
        for t in tickers
    }
    for i, day in enumerate(days):
        d = dates[i]
        day["bh_value"] = round(
            sum(bh_shares[t] * float(env.data_by_ticker[t].loc[d, "close"]) for t in tickers), 2
        )


def find_top_trades(days, n=5):
    candidates = []
    for i, day in enumerate(days):
        best_t = max(day["pnl"], key=lambda t: day["pnl"][t])
        pnl = day["pnl"][best_t]
        if pnl > 0:
            candidates.append({
                "day_idx": i,
                "date": day["date"],
                "ticker": best_t,
                "pnl": pnl,
                "action": day["actions"][best_t],
            })
    candidates.sort(key=lambda x: x["pnl"], reverse=True)
    return candidates[:n]


def main():
    if not Q_TABLE_PATH.exists():
        sys.exit(f"Q-table not found at {Q_TABLE_PATH}. Train first.")

    print("Loading Q-table...")
    agent = QLearningAgent()
    agent.load(Q_TABLE_PATH)

    print("Running test episode (2025)...")
    env, days = capture_episode(agent)
    print(f"  {len(days)} trading days")

    print("Computing buy-and-hold baseline...")
    attach_bh_curve(env, days)

    top = find_top_trades(days)
    if top:
        print(f"  Best trade: {top[0]['ticker']} +${top[0]['pnl']:.0f} on {top[0]['date']}")

    demo_data = {
        "starting_cash": STARTING_CASH,
        "tickers": sorted(env.tickers),
        "days": days,
        "top_trades": top,
    }

    out = Path(__file__).parent / "signalscope_demo.html"
    out.write_text(TEMPLATE.replace("__DEMO_DATA__", json.dumps(demo_data)), encoding="utf-8")
    print(f"\nSaved: {out}\nOpen in any browser to start the demo.")


# ---------------------------------------------------------------------------
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SignalScope — RL Trading Demo</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--panel:#0d1220;--panel2:#111827;--border:#1e2d42;
  --agent:#00e5b8;--bh:#475569;--buy:#00e5b8;--sell:#f43f5e;
  --hold:#475569;--gold:#f59e0b;--text:#e2e8f0;--muted:#64748b;
}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--text);
  font-family:'SF Mono','Fira Code',Consolas,monospace;
  display:flex;flex-direction:column}

/* HEADER */
.hdr{background:var(--panel);border-bottom:1px solid var(--border);
  padding:8px 20px;display:flex;align-items:center;gap:22px;flex-shrink:0}
.logo{font-size:14px;font-weight:700;color:var(--agent);letter-spacing:2px;white-space:nowrap}
.logo em{color:var(--muted);font-style:normal;font-weight:400}
.stats{display:flex;gap:20px;flex:1;min-width:0}
.st{display:flex;flex-direction:column;gap:1px;flex-shrink:0}
.st-l{font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.st-v{font-size:13px;font-weight:600;transition:color .2s}
.pos{color:var(--buy)}.neg{color:var(--sell)}
.ctrl{display:flex;align-items:center;gap:6px;margin-left:auto;flex-shrink:0}
.btn{background:var(--panel2);border:1px solid var(--border);color:var(--text);
  padding:5px 14px;border-radius:6px;cursor:pointer;font:inherit;font-size:12px;
  transition:all .15s;white-space:nowrap}
.btn:hover{border-color:var(--agent);color:var(--agent)}
.btn-play{width:34px;height:34px;display:flex;align-items:center;justify-content:center;
  border-radius:50%;padding:0;font-size:15px}
.btn-play.on{border-color:var(--agent);color:var(--agent);background:rgba(0,229,184,.08)}
.spd-badge{font-size:11px;color:var(--muted);min-width:30px;text-align:center}

/* MAIN */
.main{display:grid;grid-template-columns:1fr 262px;flex:1;overflow:hidden;min-height:0}

/* CHART PANEL */
.cpanel{padding:14px 16px 12px;display:flex;flex-direction:column;gap:10px;min-height:0}
.chdr{display:flex;justify-content:space-between;align-items:center}
.ctitle{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px}
.leg{display:flex;gap:14px}
.li{display:flex;align-items:center;gap:5px;font-size:9px;color:var(--muted)}
.ld{height:2px;width:16px;border-radius:1px}
.ld-a{background:var(--agent)}
.ld-b{background:var(--bh)}
.cwrap{flex:1;position:relative;min-height:0}
#chart{width:100%!important;height:100%!important}

/* CALLOUT */
.callout{position:absolute;top:16px;left:50%;transform:translateX(-50%);
  background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.5);
  border-radius:10px;padding:8px 22px;text-align:center;pointer-events:none;
  opacity:0;transition:opacity .3s;z-index:10;min-width:180px}
.callout.on{opacity:1}
.co-tag{font-size:8px;color:var(--gold);text-transform:uppercase;letter-spacing:1px}
.co-val{font-size:22px;font-weight:700;color:var(--gold);margin:2px 0}
.co-sub{font-size:9px;color:var(--muted)}

/* LOG PANEL */
.log{border-left:1px solid var(--border);background:var(--panel);
  display:flex;flex-direction:column;overflow:hidden}
.log-hdr{padding:10px 14px;border-bottom:1px solid var(--border);
  font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;
  display:flex;justify-content:space-between;align-items:center}
.log-ct{color:var(--text);font-size:10px}
.log-body{flex:1;overflow-y:auto;padding:5px}
.log-body::-webkit-scrollbar{width:3px}
.log-body::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.le{display:grid;grid-template-columns:42px 46px 38px 1fr;
  gap:3px;padding:3px 5px;border-radius:4px;font-size:10px;align-items:center}
.le:hover{background:rgba(255,255,255,.03)}
.le.new{animation:fin .2s ease}
@keyframes fin{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.ld8{color:var(--muted);font-size:9px}
.lt{font-weight:700;font-size:10px}
.la{font-size:8px;font-weight:700;padding:1px 4px;border-radius:3px;text-align:center}
.la.BUY{background:rgba(0,229,184,.14);color:var(--buy)}
.la.SELL{background:rgba(244,63,94,.14);color:var(--sell)}
.la.HOLD{color:var(--hold)}
.lp{text-align:right;font-size:9px}
.lp.p{color:var(--buy)}.lp.n{color:var(--sell)}.lp.z{color:var(--muted)}

/* TICKER STRIP */
.tstrip{border-top:1px solid var(--border);background:var(--panel);
  padding:8px 16px;display:flex;gap:8px;flex-shrink:0}
.tb{flex:1;display:flex;align-items:center;justify-content:space-between;
  padding:7px 10px;border-radius:8px;border:1px solid var(--border);
  background:var(--panel2);transition:all .25s}
.tb.BUY{border-color:var(--buy);background:rgba(0,229,184,.07);
  box-shadow:0 0 14px rgba(0,229,184,.12)}
.tb.SELL{border-color:var(--sell);background:rgba(244,63,94,.06)}
.tn{font-size:11px;font-weight:700}
.tsh{font-size:8px;color:var(--muted);margin-top:1px}
.ta{font-size:9px;font-weight:700}
.tb.BUY .ta{color:var(--buy)}.tb.SELL .ta{color:var(--sell)}.tb.HOLD .ta{color:var(--muted)}
</style>
</head>
<body>

<div class="hdr">
  <div class="logo">Signal<em>scope</em></div>
  <div class="stats">
    <div class="st"><div class="st-l">Date</div><div class="st-v" id="s-date">—</div></div>
    <div class="st"><div class="st-l">Portfolio</div><div class="st-v" id="s-pv">$10,000</div></div>
    <div class="st"><div class="st-l">Return</div><div class="st-v" id="s-ret">—</div></div>
    <div class="st"><div class="st-l">vs B&amp;H</div><div class="st-v" id="s-alpha">—</div></div>
    <div class="st"><div class="st-l">Day</div><div class="st-v" id="s-day">—</div></div>
  </div>
  <div class="ctrl">
    <button class="btn" onclick="setSpd(0)" id="btn-s0">0.5×</button>
    <button class="btn" onclick="setSpd(1)" id="btn-s1">1×</button>
    <button class="btn btn-play" id="btn-play" onclick="toggle()">▶</button>
    <button class="btn" onclick="setSpd(2)" id="btn-s2">2×</button>
    <button class="btn" onclick="restart()" style="margin-left:6px">↺</button>
  </div>
</div>

<div class="main">
  <div class="cpanel">
    <div class="chdr">
      <div class="ctitle">Portfolio Equity Curve — 2025 Out-of-Sample Eval</div>
      <div class="leg">
        <div class="li"><div class="ld ld-a"></div>RL Agent</div>
        <div class="li"><div class="ld ld-b"></div>Buy &amp; Hold</div>
        <div class="li" style="color:var(--gold)">★ Top Trade</div>
      </div>
    </div>
    <div class="cwrap">
      <canvas id="chart"></canvas>
      <div class="callout" id="callout">
        <div class="co-tag" id="co-tag">Top Trade</div>
        <div class="co-val" id="co-val">+$0</div>
        <div class="co-sub" id="co-sub"></div>
      </div>
    </div>
  </div>

  <div class="log">
    <div class="log-hdr">
      <span>Decision Log</span>
      <span class="log-ct" id="log-ct">—</span>
    </div>
    <div class="log-body" id="log-body"></div>
  </div>
</div>

<div class="tstrip" id="tstrip"></div>

<script>
const D = __DEMO_DATA__;
const SC = D.starting_cash;

// 0=0.5x, 1=1x, 2=2x  (ms per day)
const SPEEDS = [500, 200, 80];
const SLBLS  = ['0.5x','1x','2x'];
let si = 1, dayIdx = 0, tmr = null, playing = false, logN = 0;

const topSet = new Set(D.top_trades.map(t => t.day_idx));
const topMap = {};
D.top_trades.forEach(t => { topMap[t.day_idx] = t; });

// ---------- CHART ----------
const chart = new Chart(document.getElementById('chart').getContext('2d'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'RL Agent',
        data: [],
        borderColor: '#00e5b8',
        backgroundColor: 'rgba(0,229,184,0.07)',
        borderWidth: 2,
        fill: true,
        tension: 0.15,
        pointRadius: 0,
        pointHoverRadius: 5,
      },
      {
        label: 'Buy & Hold',
        data: [],
        borderColor: '#475569',
        borderDash: [5,4],
        borderWidth: 1.5,
        fill: false,
        tension: 0.15,
        pointRadius: 0,
        pointHoverRadius: 0,
      },
      {
        label: 'Stars',
        data: [],
        borderColor: 'transparent',
        backgroundColor: '#f59e0b',
        pointRadius: 10,
        pointStyle: 'star',
        showLine: false,
        pointHoverRadius: 13,
        pointHoverBackgroundColor: '#f59e0b',
      },
      {
        label: 'Cursor',
        data: [],
        borderColor: 'transparent',
        backgroundColor: '#f43f5e',
        pointRadius: 5,
        pointStyle: 'circle',
        showLine: false,
        pointHoverRadius: 5,
        order: 0,
      }
    ]
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#0d1220',
        borderColor: '#1e2d42',
        borderWidth: 1,
        titleColor: '#64748b',
        bodyColor: '#e2e8f0',
        padding: 10,
        filter: item => item.datasetIndex < 2,
        callbacks: {
          label: ctx => {
            const v = ctx.parsed.y;
            const r = ((v - SC) / SC * 100).toFixed(1);
            return ctx.dataset.label + ': $' + v.toLocaleString('en-US', {maximumFractionDigits:0}) + ' (' + (r >= 0 ? '+' : '') + r + '%)';
          }
        }
      }
    },
    scales: {
      x: {
        grid: { color: '#1a2438', tickLength: 0 },
        ticks: { color: '#4b5563', font: { size: 9 }, maxTicksLimit: 10, maxRotation: 0 }
      },
      y: {
        grid: { color: '#1a2438', tickLength: 0 },
        ticks: {
          color: '#4b5563', font: { size: 9 },
          callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'k' : v)
        }
      }
    }
  }
});

// ---------- BADGES ----------
function initBadges() {
  const s = document.getElementById('tstrip');
  D.tickers.forEach(t => {
    const el = document.createElement('div');
    el.className = 'tb HOLD';
    el.id = 'b-' + t;
    el.innerHTML = '<div><div class="tn">' + t + '</div><div class="tsh" id="sh-' + t + '">—</div></div>' +
                   '<div class="ta" id="ac-' + t + '">—</div>';
    s.appendChild(el);
  });
}

// ---------- TICK ----------
function tick() {
  if (dayIdx >= D.days.length) { stop(); return; }
  const d = D.days[dayIdx];
  const pv = d.portfolio_value, bv = d.bh_value;
  const isTop = topSet.has(dayIdx);

  // Chart update (index-assign so x-axis stays fixed at full range)
  chart.data.datasets[0].data[dayIdx] = pv;
  chart.data.datasets[1].data[dayIdx] = bv;
  chart.data.datasets[2].data[dayIdx] = isTop ? pv : null;
  // trailing red dot: clear prev, set current
  if (dayIdx > 0) chart.data.datasets[3].data[dayIdx - 1] = null;
  chart.data.datasets[3].data[dayIdx] = pv;
  chart.update('none');

  // Header stats
  const ret   = (pv - SC) / SC * 100;
  const bret  = (bv - SC) / SC * 100;
  const alpha = ret - bret;
  document.getElementById('s-date').textContent = d.date;
  document.getElementById('s-pv').textContent = '$' + pv.toLocaleString('en-US', {maximumFractionDigits:0});
  setVal('s-ret',   (ret>=0?'+':'')   + ret.toFixed(1)   + '%',   ret >= 0);
  setVal('s-alpha', (alpha>=0?'+':'') + alpha.toFixed(1) + ' pp', alpha >= 0);
  document.getElementById('s-day').textContent = (dayIdx+1) + ' / ' + D.days.length;

  // Ticker badges
  D.tickers.forEach(t => {
    const act = d.actions[t];
    document.getElementById('b-' + t).className = 'tb ' + act;
    document.getElementById('ac-' + t).textContent = act;
    document.getElementById('sh-' + t).textContent = d.positions[t] + ' sh';
  });

  // Decision log (show non-HOLD; if all hold, show one)
  const active = D.tickers.filter(t => d.actions[t] !== 'HOLD');
  const toLog  = active.length ? active : [D.tickers[0]];
  const lb = document.getElementById('log-body');
  toLog.slice().reverse().forEach(t => {
    const act = d.actions[t];
    const pnl = d.pnl[t];
    const ps  = Math.abs(pnl) > 0.5 ? (pnl>0?'+':'') + '$' + Math.abs(pnl).toFixed(0) : '—';
    const pc  = pnl > 0.5 ? 'p' : pnl < -0.5 ? 'n' : 'z';
    const e = document.createElement('div');
    e.className = 'le new';
    e.innerHTML = '<span class="ld8">' + d.date.slice(5) + '</span>' +
      '<span class="lt">' + t + '</span>' +
      '<span class="la ' + act + '">' + act + '</span>' +
      '<span class="lp ' + pc + '">' + ps + '</span>';
    lb.insertBefore(e, lb.firstChild);
    logN++;
  });
  document.getElementById('log-ct').textContent = logN;

  // Callout for top trade
  if (isTop) {
    const tr = topMap[dayIdx];
    document.getElementById('co-tag').textContent = tr.ticker + ' — Best Trade';
    document.getElementById('co-val').textContent = '+$' + tr.pnl.toFixed(0);
    document.getElementById('co-sub').textContent = tr.date;
    const c = document.getElementById('callout');
    c.classList.add('on');
    setTimeout(() => c.classList.remove('on'), 2500);
  }

  dayIdx++;
}

function setVal(id, txt, isPos) {
  const e = document.getElementById(id);
  e.textContent = txt;
  e.className = 'st-v ' + (isPos ? 'pos' : 'neg');
}

// ---------- CONTROLS ----------
function stop() {
  clearInterval(tmr); tmr = null; playing = false;
  const b = document.getElementById('btn-play');
  b.textContent = '▶'; b.classList.remove('on');
}

function toggle() {
  if (playing) {
    stop();
  } else {
    if (dayIdx >= D.days.length) restart();
    playing = true;
    document.getElementById('btn-play').textContent = '⏸';
    document.getElementById('btn-play').classList.add('on');
    tmr = setInterval(tick, SPEEDS[si]);
  }
}

function setSpd(idx) {
  si = idx;
  ['btn-s0','btn-s1','btn-s2'].forEach((id, i) => {
    document.getElementById(id).style.borderColor = i === idx ? 'var(--agent)' : '';
    document.getElementById(id).style.color = i === idx ? 'var(--agent)' : '';
  });
  if (playing) { clearInterval(tmr); tmr = setInterval(tick, SPEEDS[si]); }
}

function restart() {
  stop(); dayIdx = 0; logN = 0;
  const nulls = () => D.days.map(() => null);
  chart.data.labels = D.days.map(d => d.date.slice(5));
  chart.data.datasets[0].data = nulls();
  chart.data.datasets[1].data = nulls();
  chart.data.datasets[2].data = nulls();
  chart.data.datasets[3].data = nulls();
  chart.update('none');
  document.getElementById('log-body').innerHTML = '';
  document.getElementById('log-ct').textContent = '—';
  document.getElementById('s-date').textContent = '—';
  document.getElementById('s-pv').textContent = '$10,000';
  ['s-ret','s-alpha'].forEach(id => {
    const e = document.getElementById(id);
    e.textContent = '—'; e.className = 'st-v';
  });
  document.getElementById('s-day').textContent = '— / ' + D.days.length;
  D.tickers.forEach(t => {
    document.getElementById('b-' + t).className = 'tb HOLD';
    document.getElementById('ac-' + t).textContent = '—';
    document.getElementById('sh-' + t).textContent = '—';
  });
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space')      { e.preventDefault(); toggle(); }
  else if (e.code === 'KeyR')  restart();
  else if (e.code === 'Digit1') setSpd(0);
  else if (e.code === 'Digit2') setSpd(1);
  else if (e.code === 'Digit3') setSpd(2);
});

initBadges();
restart();
</script>
</body>
</html>"""

if __name__ == "__main__":
    main()
