# =============================================================================
# FILE: server.jl
# PURPOSE: Main HTTP API Server — Entry point for the backend
#          Wires together all modules (model, betting, monte_carlo)
#          and exposes a REST API consumed by the HTML/JS frontend
#
# HOW TO RUN:
#   cd backend/
#   julia --project=. server.jl
#   (First run will install packages + train model — takes ~5-10 min)
#
# API ENDPOINTS:
#   GET  /api/health          → Server status + model readiness
#   GET  /api/teams           → List of all available teams
#   POST /api/predict         → XGBoost match prediction + confidence intervals
#   POST /api/betting-ev      → Betting expected value analysis
#   POST /api/monte-carlo     → Monte Carlo season simulator
#   GET  /api/team-stats/:team → Historical stats for a specific team
#   GET  /api/head-to-head    → H2H record between two teams
# =============================================================================

# --- Package setup ---
using Pkg
Pkg.activate(dirname(@__FILE__))

using HTTP, JSON3, Logging, Dates, DataFrames, Statistics

# Load local modules
include("features.jl")
include("model.jl")
include("betting.jl")
include("monte_carlo.jl")

using .Features, .Model, .Betting, .MonteCarlo

# =============================================================================
# CONFIGURATION
# =============================================================================
# Read HOST and PORT from environment — required by Render and Railway
# Locally defaults to 127.0.0.1:8080
const HOST = get(ENV, "HOST", "0.0.0.0")          # 0.0.0.0 = accept external connections
const PORT = parse(Int, get(ENV, "PORT", "8080"))
const DATA_PATH = joinpath(@__DIR__, "data", "epl_final.csv")
const CORS_HEADERS = [
    "Access-Control-Allow-Origin"  => "*",
    "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers" => "Content-Type, Accept",
    "Content-Type"                 => "application/json"
]

# =============================================================================
# HELPERS
# =============================================================================
json_response(data; status=200) = HTTP.Response(
    status,
    CORS_HEADERS,
    body = JSON3.write(data)
)

error_response(msg::String; status=400) = json_response(
    Dict("error" => msg, "status" => status);
    status=status
)

function parse_json_body(req::HTTP.Request)
    isempty(req.body) && return Dict()
    JSON3.read(String(req.body), Dict)
end

# =============================================================================
# ROUTE HANDLERS
# =============================================================================

# GET /api/health
function handle_health(req)
    json_response(Dict(
        "status"    => "ok",
        "model"     => Model.STATE.is_trained ? "ready" : "training",
        "teams"     => length(Model.STATE.teams),
        "timestamp" => string(now())
    ))
end

# GET /api/teams
function handle_teams(req)
    !Model.STATE.is_trained && return error_response("Model not ready", status=503)
    json_response(Dict(
        "teams"       => Model.STATE.teams,
        "count"       => length(Model.STATE.teams),
        "elo_ratings" => Dict(t => round(Model.STATE.current_elo[t], digits=1)
                              for t in Model.STATE.teams if haskey(Model.STATE.current_elo, t))
    ))
end

# POST /api/predict
# Body: { "home": "Arsenal", "away": "Chelsea", "confidence": true }
function handle_predict(req)
    !Model.STATE.is_trained && return error_response("Model not ready — still training", status=503)

    body = parse_json_body(req)
    home = get(body, "home", "")
    away = get(body, "away", "")
    use_ci = get(body, "confidence", true)

    isempty(home) && return error_response("Missing 'home' team")
    isempty(away) && return error_response("Missing 'away' team")
    home == away  && return error_response("Home and away teams must differ")

    home ∉ Model.STATE.teams && return error_response("Unknown home team: $home")
    away ∉ Model.STATE.teams && return error_response("Unknown away team: $away")

    try
        if use_ci
            r = Model.predict_with_confidence(home, away, n_bootstrap=300)
            json_response(Dict(
                "home_team"  => home,
                "away_team"  => away,
                "home_win"   => round(r.home_win, digits=4),
                "draw"       => round(r.draw, digits=4),
                "away_win"   => round(r.away_win, digits=4),
                "confidence_intervals" => Dict(
                    "home_win" => Dict("low" => round(r.home_win_ci[1], digits=4),
                                       "high" => round(r.home_win_ci[2], digits=4)),
                    "draw"     => Dict("low" => round(r.draw_ci[1], digits=4),
                                       "high" => round(r.draw_ci[2], digits=4)),
                    "away_win" => Dict("low" => round(r.away_win_ci[1], digits=4),
                                       "high" => round(r.away_win_ci[2], digits=4)),
                ),
                "n_bootstrap"=> r.n_bootstrap,
                "note"       => r.note,
                "predicted_result" => r.home_win > r.away_win && r.home_win > r.draw ? "Home Win" :
                                      r.away_win > r.home_win && r.away_win > r.draw ? "Away Win" : "Draw"
            ))
        else
            r = Model.predict_match(home, away)
            json_response(Dict(
                "home_team" => home,
                "away_team" => away,
                "home_win"  => round(r.home_win, digits=4),
                "draw"      => round(r.draw, digits=4),
                "away_win"  => round(r.away_win, digits=4),
                "predicted_result" => r.home_win > r.away_win && r.home_win > r.draw ? "Home Win" :
                                      r.away_win > r.home_win && r.away_win > r.draw ? "Away Win" : "Draw"
            ))
        end
    catch e
        @error "Prediction error" exception=e
        error_response("Prediction failed: $(string(e))", status=500)
    end
end

# POST /api/betting-ev
# Body: {
#   "home": "Arsenal", "away": "Chelsea",
#   "odds": { "home": 2.10, "draw": 3.50, "away": 3.80 }
# }
function handle_betting_ev(req)
    !Model.STATE.is_trained && return error_response("Model not ready", status=503)

    body = parse_json_body(req)
    home = get(body, "home", "")
    away = get(body, "away", "")
    odds = get(body, "odds", Dict())

    isempty(home) && return error_response("Missing 'home'")
    isempty(away) && return error_response("Missing 'away'")
    isempty(odds) && return error_response("Missing 'odds' {home, draw, away}")

    book_home = Float64(get(odds, "home", 0.0))
    book_draw = Float64(get(odds, "draw", 0.0))
    book_away = Float64(get(odds, "away", 0.0))

    (book_home <= 1.0 || book_draw <= 1.0 || book_away <= 1.0) &&
        return error_response("All odds must be decimal > 1.0")

    try
        p = Model.predict_match(home, away)
        result = Betting.compute_betting_ev(
            p.home_win, p.draw, p.away_win,
            book_home, book_draw, book_away
        )

        json_response(Dict(
            "home_team" => home,
            "away_team" => away,
            "model_probabilities" => Dict(
                "home_win" => round(p.home_win, digits=4),
                "draw"     => round(p.draw, digits=4),
                "away_win" => round(p.away_win, digits=4),
            ),
            "book_odds" => Dict("home" => book_home, "draw" => book_draw, "away" => book_away),
            "overround" => round(result.overround, digits=4),
            "vig_pct"   => round(result.vig_pct, digits=2),
            "has_value" => result.has_value,
            "analyses"  => [Dict(
                "outcome"       => a.outcome,
                "model_prob"    => round(a.model_prob, digits=4),
                "book_odds"     => a.book_odds,
                "implied_prob"  => round(a.implied_prob, digits=4),
                "no_vig_prob"   => round(a.no_vig_prob, digits=4),
                "expected_value"=> round(a.expected_value, digits=4),
                "kelly_fraction"=> round(a.kelly_fraction, digits=4),
                "is_value_bet"  => a.is_value_bet,
                "edge"          => round(a.edge, digits=4)
            ) for a in result.analyses],
            "best_bet" => Dict(
                "outcome"        => result.best_bet.outcome,
                "expected_value" => round(result.best_bet.expected_value, digits=4),
                "kelly_fraction" => round(result.best_bet.kelly_fraction, digits=4),
                "is_value"       => result.best_bet.is_value_bet
            )
        ))
    catch e
        @error "Betting EV error" exception=e
        error_response("Betting analysis failed: $(string(e))", status=500)
    end
end

# POST /api/monte-carlo
# Body: { "n_sims": 5000, "teams": [...optional override...] }
function handle_monte_carlo(req)
    !Model.STATE.is_trained && return error_response("Model not ready", status=503)

    body   = parse_json_body(req)
    n_sims = Int(get(body, "n_sims", 3000))
    n_sims = clamp(n_sims, 100, 10000)

    custom_teams = get(body, "teams", nothing)
    teams = isnothing(custom_teams) ?
        MonteCarlo.CURRENT_EPL_TEAMS :
        filter(t -> t in Model.STATE.teams, String.(custom_teams))

    isempty(teams) && return error_response("No valid teams provided")

    try
        @info "Starting Monte Carlo simulation: $n_sims × $(length(teams)) teams"
        results = MonteCarlo.simulate_season(
            Model.predict_match;
            teams = teams,
            n_sims = n_sims
        )

        json_response(Dict(
            "n_simulations" => n_sims,
            "n_teams"       => length(teams),
            "standings"     => [Dict(
                "position"        => i,
                "team"            => r.team,
                "mean_points"     => round(r.mean_points, digits=1),
                "std_points"      => round(r.std_points, digits=1),
                "ci_low"          => round(r.ci_low, digits=0),
                "ci_high"         => round(r.ci_high, digits=0),
                "mean_position"   => round(r.mean_position, digits=1),
                "prob_title"      => round(r.prob_title * 100, digits=1),
                "prob_top4"       => round(r.prob_top4 * 100, digits=1),
                "prob_top6"       => round(r.prob_top6 * 100, digits=1),
                "prob_relegation" => round(r.prob_relegation * 100, digits=1),
            ) for (i, r) in enumerate(results)]
        ))
    catch e
        @error "Monte Carlo error" exception=e
        error_response("Simulation failed: $(string(e))", status=500)
    end
end

# GET /api/team-stats?team=Arsenal
function handle_team_stats(req)
    !Model.STATE.is_trained && return error_response("Model not ready", status=503)

    params = HTTP.queryparams(HTTP.URI(req.target))
    team   = get(params, "team", "")
    isempty(team) && return error_response("Missing ?team= parameter")
    team ∉ Model.STATE.teams && return error_response("Unknown team: $team")

    s = Features.get_team_stats(Model.STATE.df, team, Model.STATE.current_elo)

    json_response(Dict(
        "team"                  => team,
        "elo"                   => round(s.elo, digits=1),
        "home_win_rate"         => round(s.home_win_rate * 100, digits=1),
        "away_win_rate"         => round(s.away_win_rate * 100, digits=1),
        "draw_rate"             => round(s.draw_rate * 100, digits=1),
        "avg_goals_scored_home" => round(s.avg_goals_scored_home, digits=2),
        "avg_goals_scored_away" => round(s.avg_goals_scored_away, digits=2),
        "avg_goals_conceded_home"=> round(s.avg_goals_conceded_home, digits=2),
        "avg_goals_conceded_away"=> round(s.avg_goals_conceded_away, digits=2),
        "avg_sot_home"          => round(s.avg_home_sot, digits=2),
        "avg_sot_away"          => round(s.avg_away_sot, digits=2),
        "home_recent_goals"     => round(s.home_recent_goals, digits=2),
        "away_recent_goals"     => round(s.away_recent_goals, digits=2),
        "total_games"           => s.total_home_games + s.total_away_games,
    ))
end

# GET /api/head-to-head?home=Arsenal&away=Chelsea
function handle_h2h(req)
    !Model.STATE.is_trained && return error_response("Model not ready", status=503)

    params = HTTP.queryparams(HTTP.URI(req.target))
    home   = get(params, "home", "")
    away   = get(params, "away", "")
    isempty(home) && return error_response("Missing ?home= parameter")
    isempty(away) && return error_response("Missing ?away= parameter")

    df = Model.STATE.df

    # All H2H matches regardless of home/away assignment
    h2h = df[
        ((df.HomeTeam .== home) .& (df.AwayTeam .== away)) .|
        ((df.HomeTeam .== away) .& (df.AwayTeam .== home)),
    :]

    nrow(h2h) == 0 && return error_response("No head-to-head data found")

    home_wins = sum((h2h.HomeTeam .== home .& h2h.FullTimeResult .== "H") .||
                    (h2h.AwayTeam .== home .& h2h.FullTimeResult .== "A"))
    away_wins = sum((h2h.HomeTeam .== away .& h2h.FullTimeResult .== "H") .||
                    (h2h.AwayTeam .== away .& h2h.FullTimeResult .== "A"))
    draws     = nrow(h2h) - home_wins - away_wins

    recent = last(sort(h2h, :MatchDate), min(5, nrow(h2h)))
    recent_results = [Dict(
        "date"   => string(r.MatchDate),
        "home"   => r.HomeTeam,
        "away"   => r.AwayTeam,
        "score"  => "$(r.FullTimeHomeGoals)-$(r.FullTimeAwayGoals)",
        "result" => r.FullTimeResult
    ) for r in eachrow(recent)]

    json_response(Dict(
        "team_a"         => home,
        "team_b"         => away,
        "total_meetings" => nrow(h2h),
        "team_a_wins"    => home_wins,
        "team_b_wins"    => away_wins,
        "draws"          => draws,
        "recent_meetings"=> recent_results
    ))
end

# =============================================================================
# ROUTER
# =============================================================================
function router(req::HTTP.Request)
    # Handle CORS preflight
    req.method == "OPTIONS" && return HTTP.Response(204, CORS_HEADERS)

    path = HTTP.URI(req.target).path

    try
        if     path == "/api/health"       return handle_health(req)
        elseif path == "/api/teams"        return handle_teams(req)
        elseif path == "/api/predict"      return handle_predict(req)
        elseif path == "/api/betting-ev"   return handle_betting_ev(req)
        elseif path == "/api/monte-carlo"  return handle_monte_carlo(req)
        elseif path == "/api/team-stats"   return handle_team_stats(req)
        elseif path == "/api/head-to-head" return handle_h2h(req)
        else
            return error_response("Route not found: $path", status=404)
        end
    catch e
        @error "Unhandled server error" path=path exception=e
        return error_response("Internal server error", status=500)
    end
end

# =============================================================================
# STARTUP
# =============================================================================
function main()
    @info "=== EPL Prediction Engine — Julia Backend ==="
    @info "Loading/training model from $DATA_PATH ..."

    load_or_train_model(DATA_PATH)

    @info "Starting HTTP server on http://$HOST:$PORT"
    @info "Open frontend/index.html in your browser to use the UI."

    HTTP.serve(router, HOST, PORT; verbose=false)
end

main()
