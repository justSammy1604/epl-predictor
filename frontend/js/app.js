// =============================================================================
// FILE: frontend/js/app.js
// PURPOSE: Main frontend application — UI state, event listeners,
//          rendering prediction/betting/MC results from API responses
// IMPORTS: api.js (all backend calls), charts.js (Chart.js wrappers)
// =============================================================================

import * as API    from "./api.js";
import * as Charts from "./charts.js";

// =============================================================================
// STATE
// =============================================================================
let TEAMS     = [];
let serverOK  = false;

// =============================================================================
// INIT
// =============================================================================
document.addEventListener("DOMContentLoaded", async () => {
  await checkServer();
  setupTabs();
  setupPredictTab();
  setupBettingTab();
  setupMonteCarloTab();
});

// =============================================================================
// SERVER HEALTH POLLING
// =============================================================================
async function checkServer() {
  const dot   = document.getElementById("statusDot");
  const label = document.getElementById("statusLabel");
  try {
    const h = await API.checkHealth();
    if (h.model === "ready") {
      serverOK = true;
      dot.className   = "status-dot online";
      label.textContent = `Connected · ${h.teams} teams · Model Ready`;
      await loadTeams();
    } else {
      dot.className   = "status-dot training";
      label.textContent = "Model training… please wait";
      setTimeout(checkServer, 5000);
    }
  } catch {
    dot.className   = "status-dot offline";
    label.textContent = "Backend offline — start server.jl";
    setTimeout(checkServer, 8000);
  }
}

async function loadTeams() {
  const data = await API.getTeams();
  TEAMS = data.teams;
  populateSelects(["homeTeam", "awayTeam", "betHome", "betAway"]);
  // Render ELO table
  renderEloTable(data.elo_ratings);
}

function populateSelects(ids) {
  ids.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = `<option value="">— Select team —</option>` +
      TEAMS.map(t => `<option value="${t}"${t === prev ? " selected" : ""}>${t}</option>`).join("");
  });
}

function renderEloTable(ratings) {
  const tbody = document.getElementById("eloBody");
  if (!tbody || !ratings) return;
  const sorted = Object.entries(ratings).sort((a,b) => b[1] - a[1]);
  tbody.innerHTML = sorted.map(([team, elo], i) => `
    <tr>
      <td class="pos">${i + 1}</td>
      <td>${team}</td>
      <td class="elo-val">${elo.toFixed(0)}</td>
      <td><div class="elo-bar" style="width:${((elo-1200)/800*100).toFixed(0)}%"></div></td>
    </tr>`).join("");
}

// =============================================================================
// TABS
// =============================================================================
function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });
}

// =============================================================================
// PREDICTION TAB
// =============================================================================
function setupPredictTab() {
  document.getElementById("predictBtn").addEventListener("click", runPrediction);
  document.getElementById("h2hBtn").addEventListener("click", loadH2H);
  document.getElementById("homeTeam").addEventListener("change", loadTeamCard.bind(null, "home"));
  document.getElementById("awayTeam").addEventListener("change", loadTeamCard.bind(null, "away"));
}

async function runPrediction() {
  if (!serverOK) return toast("Backend offline", "error");

  const home = document.getElementById("homeTeam").value;
  const away = document.getElementById("awayTeam").value;
  if (!home || !away) return toast("Select both teams", "warning");
  if (home === away)  return toast("Teams must differ", "warning");

  const btn = document.getElementById("predictBtn");
  setLoading(btn, true);

  try {
    const data = await API.predictMatch(home, away, true);
    renderPrediction(data);
    Charts.renderPredictionChart(data);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    setLoading(btn, false);
  }
}

function renderPrediction(d) {
  document.getElementById("predictionPanel").classList.remove("hidden");

  const winner = d.home_win > d.away_win && d.home_win > d.draw ? d.home_team :
                 d.away_win > d.home_win && d.away_win > d.draw ? d.away_team : "Draw";

  const ci = d.confidence_intervals;

  document.getElementById("predResult").innerHTML = `
    <div class="result-badge">${winner === "Draw" ? "⚖️ Draw" : "🏆 " + winner}</div>
    <div class="pred-probs">
      ${probCard(d.home_team, d.home_win, ci?.home_win, "home")}
      ${probCard("Draw",      d.draw,     ci?.draw,     "draw")}
      ${probCard(d.away_team, d.away_win, ci?.away_win, "away")}
    </div>
    <div class="meta-row">
      <span>ELO: <b>${d.home_team}</b> ${d.elo_home?.toFixed(0)} vs <b>${d.away_team}</b> ${d.elo_away?.toFixed(0)}</span>
      <span>Form (recent goals): ${d.home_form?.toFixed(2)} vs ${d.away_form?.toFixed(2)}</span>
      ${d.n_bootstrap ? `<span class="badge">Bootstrap CI n=${d.n_bootstrap}</span>` : ""}
    </div>`;
}

function probCard(label, prob, ci, type) {
  const pct = (prob * 100).toFixed(1);
  const ciHtml = ci
    ? `<span class="ci">[${(ci.low*100).toFixed(1)}% – ${(ci.high*100).toFixed(1)}%]</span>`
    : "";
  return `<div class="prob-card ${type}">
    <div class="prob-pct">${pct}%</div>
    <div class="prob-label">${label}</div>
    ${ciHtml}
    <div class="prob-fill" style="height:${pct}%"></div>
  </div>`;
}

async function loadH2H() {
  const home = document.getElementById("homeTeam").value;
  const away = document.getElementById("awayTeam").value;
  if (!home || !away) return;

  try {
    const d = await API.getH2H(home, away);
    const box = document.getElementById("h2hBox");
    box.classList.remove("hidden");
    box.innerHTML = `
      <h4>Head-to-Head: ${d.team_a} vs ${d.team_b}</h4>
      <div class="h2h-summary">
        <div class="h2h-stat"><span>${d.team_a_wins}</span> Wins for ${d.team_a}</div>
        <div class="h2h-stat draw"><span>${d.draws}</span> Draws</div>
        <div class="h2h-stat away"><span>${d.team_b_wins}</span> Wins for ${d.team_b}</div>
      </div>
      <p class="h2h-total">${d.total_meetings} total meetings in dataset</p>
      <h5>Last 5 Meetings</h5>
      <table class="h2h-table">
        <thead><tr><th>Date</th><th>Home</th><th>Score</th><th>Away</th></tr></thead>
        <tbody>${d.recent_meetings.map(m => `
          <tr>
            <td>${m.date}</td>
            <td class="${m.result==="H"?"win":""}">${m.home}</td>
            <td class="score">${m.score}</td>
            <td class="${m.result==="A"?"win":""}">${m.away}</td>
          </tr>`).join("")}
        </tbody>
      </table>`;
  } catch (e) {
    toast(e.message, "error");
  }
}

async function loadTeamCard(side) {
  const team = document.getElementById(side === "home" ? "homeTeam" : "awayTeam").value;
  if (!team) return;
  try {
    const s = await API.getTeamStats(team);
    const id = side + "Card";
    document.getElementById(id).innerHTML = `
      <h4>${team}</h4>
      <div class="stat-grid">
        <div class="stat"><label>ELO</label><b>${s.elo}</b></div>
        <div class="stat"><label>Home Win%</label><b>${s.home_win_rate}%</b></div>
        <div class="stat"><label>Away Win%</label><b>${s.away_win_rate}%</b></div>
        <div class="stat"><label>Draw%</label><b>${s.draw_rate}%</b></div>
        <div class="stat"><label>Avg Goals H</label><b>${s.avg_goals_scored_home}</b></div>
        <div class="stat"><label>Avg Goals A</label><b>${s.avg_goals_scored_away}</b></div>
        <div class="stat"><label>Avg SOT H</label><b>${s.avg_sot_home}</b></div>
        <div class="stat"><label>Games</label><b>${s.total_games}</b></div>
      </div>`;
  } catch {}
}

// =============================================================================
// BETTING EV TAB
// =============================================================================
function setupBettingTab() {
  document.getElementById("betBtn").addEventListener("click", runBettingEV);
}

async function runBettingEV() {
  if (!serverOK) return toast("Backend offline", "error");

  const home     = document.getElementById("betHome").value;
  const away     = document.getElementById("betAway").value;
  const oddsHome = parseFloat(document.getElementById("oddsHome").value);
  const oddsDraw = parseFloat(document.getElementById("oddsDraw").value);
  const oddsAway = parseFloat(document.getElementById("oddsAway").value);

  if (!home || !away)     return toast("Select both teams", "warning");
  if ([oddsHome, oddsDraw, oddsAway].some(v => isNaN(v) || v <= 1))
    return toast("All decimal odds must be > 1.0", "warning");

  const btn = document.getElementById("betBtn");
  setLoading(btn, true);

  try {
    const data = await API.getBettingEV(home, away, {
      home: oddsHome, draw: oddsDraw, away: oddsAway
    });
    renderBettingEV(data);
    Charts.renderEVChart(data);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    setLoading(btn, false);
  }
}

function renderBettingEV(d) {
  document.getElementById("evPanel").classList.remove("hidden");

  const vigLabel = `Bookmaker Margin: ${d.vig_pct.toFixed(2)}%`;
  const valueBets = d.analyses.filter(a => a.is_value_bet);

  document.getElementById("evResult").innerHTML = `
    <div class="ev-summary">
      <div class="ev-card ${d.has_value ? "value" : "no-value"}">
        ${d.has_value ? "✅ Value Bet Found!" : "❌ No Value Detected"}
      </div>
      <span class="badge">${vigLabel}</span>
    </div>
    <table class="ev-table">
      <thead>
        <tr><th>Outcome</th><th>Model %</th><th>Book Odds</th><th>No-Vig %</th><th>EV</th><th>Kelly</th><th>Edge</th></tr>
      </thead>
      <tbody>
        ${d.analyses.map(a => `
          <tr class="${a.is_value_bet ? "value-row" : ""}">
            <td>${a.outcome}</td>
            <td>${(a.model_prob*100).toFixed(1)}%</td>
            <td>${a.book_odds.toFixed(2)}</td>
            <td>${(a.no_vig_prob*100).toFixed(1)}%</td>
            <td class="${a.expected_value > 0 ? "positive" : "negative"}">${(a.expected_value*100).toFixed(2)}%</td>
            <td>${a.kelly_fraction > 0 ? (a.kelly_fraction*100).toFixed(1)+"%" : "—"}</td>
            <td class="${a.edge > 0 ? "positive" : "negative"}">${(a.edge*100).toFixed(1)}%</td>
          </tr>`).join("")}
      </tbody>
    </table>
    ${valueBets.length ? `
    <div class="kelly-advice">
      <h4>📐 Kelly Staking Recommendation</h4>
      ${valueBets.map(a => `
        <p><b>${a.outcome}</b> — Stake <b>${(a.kelly_fraction*100).toFixed(1)}%</b> of bankroll
        (EV: +${(a.expected_value*100).toFixed(2)}% per £1 staked)</p>`).join("")}
      <small>Half-Kelly applied. Maximum 25% of bankroll per bet.</small>
    </div>` : ""}`;
}

// =============================================================================
// MONTE CARLO TAB
// =============================================================================
function setupMonteCarloTab() {
  document.getElementById("mcBtn").addEventListener("click", runMonteCarlo);
}

async function runMonteCarlo() {
  if (!serverOK) return toast("Backend offline", "error");

  const nSims = parseInt(document.getElementById("nSims").value) || 3000;
  const btn   = document.getElementById("mcBtn");
  setLoading(btn, true, `Simulating ${nSims.toLocaleString()} seasons…`);

  try {
    const data = await API.runMonteCarlo(nSims);
    renderMCStandings(data);
    Charts.renderMonteCarloChart(data);
  } catch (e) {
    toast(e.message, "error");
  } finally {
    setLoading(btn, false);
  }
}

function renderMCStandings(data) {
  document.getElementById("mcPanel").classList.remove("hidden");
  document.getElementById("mcMeta").textContent =
    `${data.n_simulations.toLocaleString()} simulations · ${data.n_teams} teams`;

  const tbody = document.getElementById("mcBody");
  tbody.innerHTML = data.standings.map(r => `
    <tr>
      <td class="pos">${r.position}</td>
      <td class="team-name">${r.team}</td>
      <td>${r.mean_points.toFixed(0)} <span class="ci-tiny">±${r.std_points.toFixed(0)}</span></td>
      <td><span class="ci-range">${r.ci_low.toFixed(0)}–${r.ci_high.toFixed(0)}</span></td>
      <td>${r.mean_position.toFixed(1)}</td>
      <td class="prob-cell title">${r.prob_title.toFixed(1)}%</td>
      <td class="prob-cell cl">${r.prob_top4.toFixed(1)}%</td>
      <td class="prob-cell el">${r.prob_top6.toFixed(1)}%</td>
      <td class="prob-cell rel">${r.prob_relegation.toFixed(1)}%</td>
    </tr>`).join("");
}

// =============================================================================
// UTILITIES
// =============================================================================
function setLoading(btn, loading, text = null) {
  btn.disabled = loading;
  if (loading) {
    btn.dataset.origText = btn.textContent;
    btn.innerHTML = `<span class="spinner"></span> ${text || "Loading…"}`;
  } else {
    btn.innerHTML = btn.dataset.origText || "Submit";
  }
}

function toast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}
