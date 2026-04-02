// =============================================================================
// FILE: frontend/js/api.js
// PURPOSE: All HTTP calls to the Julia backend
//
// LOCAL:      points to http://127.0.0.1:8080/api  (default)
// PRODUCTION: set window.EPL_API_BASE in index.html before this script loads
//             e.g. <script>window.EPL_API_BASE = "https://your-app.onrender.com/api"</script>
// =============================================================================

const API_BASE = window.EPL_API_BASE || "http://127.0.0.1:8080/api";

// Generic fetch wrapper with error handling
async function apiCall(path, method = "GET", body = null) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);

  const res = await fetch(`${API_BASE}${path}`, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// --- API Functions ---

/** Check if the Julia server and model are ready */
export async function checkHealth() {
  return apiCall("/health");
}

/** Get full list of teams + ELO ratings */
export async function getTeams() {
  return apiCall("/teams");
}

/**
 * Get match prediction with optional 95% bootstrap confidence intervals
 * @param {string} home
 * @param {string} away
 * @param {boolean} withConfidence - Whether to run bootstrap CI (slower)
 */
export async function predictMatch(home, away, withConfidence = true) {
  return apiCall("/predict", "POST", {
    home,
    away,
    confidence: withConfidence,
  });
}

/**
 * Get betting expected value analysis
 * @param {string} home
 * @param {string} away
 * @param {{ home: number, draw: number, away: number }} odds - Decimal odds
 */
export async function getBettingEV(home, away, odds) {
  return apiCall("/betting-ev", "POST", { home, away, odds });
}

/**
 * Run Monte Carlo season simulator
 * @param {number} nSims - Number of simulations (100-10000)
 * @param {string[]} teams - Optional list of teams to simulate
 */
export async function runMonteCarlo(nSims = 3000, teams = null) {
  const body = { n_sims: nSims };
  if (teams) body.teams = teams;
  return apiCall("/monte-carlo", "POST", body);
}

/** Get historical stats for a single team */
export async function getTeamStats(team) {
  return apiCall(`/team-stats?team=${encodeURIComponent(team)}`);
}

/** Get head-to-head record between two teams */
export async function getH2H(home, away) {
  return apiCall(
    `/head-to-head?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`
  );
}
