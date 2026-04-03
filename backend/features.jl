# =============================================================================
# FILE: features.jl
# PURPOSE: Feature Engineering — ELO Ratings, Rolling Averages, Team Stats
# USED BY: model.jl, monte_carlo.jl
# =============================================================================

module Features

using DataFrames, Statistics, StatsBase, Dates

export compute_elo!, compute_yellow_cards!, build_feature_matrix,
       get_team_stats, get_all_team_stats, FEATURE_COLS

# Features the XGBoost model is trained on — must match model.jl exactly
const FEATURE_COLS = [
    :AwayShotsOnTarget, :HomeShotsOnTarget,
    :HalfTimeHomeGoals, :HalfTimeAwayGoals,
    :HomeYellowCards, :AwayYellowCards,
    :HomeCorners, :AwayCorners
]

# -----------------------------------------------------------------------------
# ELO Rating System
# Updates ELO in-place after each chronological match.
# K=20 (standard football ELO K-factor)
# Starting rating = 1500 for all teams
# -----------------------------------------------------------------------------
function compute_elo!(df::DataFrame; K::Float64 = 20.0, start_rating::Float64 = 1500.0)
    teams = unique(vcat(df.HomeTeam, df.AwayTeam))
    elo = Dict(team => start_rating for team in teams)

    sort!(df, :MatchDate)
    df.EloHome = zeros(Float64, nrow(df))
    df.EloAway = zeros(Float64, nrow(df))

    for i in 1:nrow(df)
        home = df.HomeTeam[i]
        away = df.AwayTeam[i]

        Rh = elo[home]
        Ra = elo[away]
        df.EloHome[i] = Rh
        df.EloAway[i] = Ra

        # Expected win probability via ELO formula
        Eh = 1.0 / (1.0 + 10.0 ^ ((Ra - Rh) / 400.0))

        # Actual score (1=home win, 0.5=draw, 0=away win)
        S = df.FullTimeResult[i] == "H" ? 1.0 :
            df.FullTimeResult[i] == "D" ? 0.5 : 0.0

        # Update both ELOs
        elo[home] += K * (S - Eh)
        elo[away] += K * ((1.0 - S) - (1.0 - Eh))
    end

    return elo  # Return final ELO dictionary for live predictions
end

# -----------------------------------------------------------------------------
# Yellow Cards Computation
# Computes average yellow cards per team for home and away games
# Used as a feature for match predictions
# -----------------------------------------------------------------------------
function compute_yellow_cards!(df::DataFrame; window::Int = 5)
    df.HomeYellowCards = zeros(Float64, nrow(df))
    df.AwayYellowCards = zeros(Float64, nrow(df))

    for i in 1:nrow(df)
        home_team = df.HomeTeam[i]
        away_team = df.AwayTeam[i]
        prev = df[1:i-1, :]

        # Home yellow cards (when team plays at home)
        home_games = prev[prev.HomeTeam .== home_team, :]
        if nrow(home_games) > 0
            n = min(window, nrow(home_games))
            df.HomeYellowCards[i] = mean(last(home_games.HomeYellowCards, n))
        else
            df.HomeYellowCards[i] = 2.0  # Default EPL average
        end

        # Away yellow cards (when team plays away)
        away_games = prev[prev.AwayTeam .== away_team, :]
        if nrow(away_games) > 0
            n = min(window, nrow(away_games))
            df.AwayYellowCards[i] = mean(last(away_games.AwayYellowCards, n))
        else
            df.AwayYellowCards[i] = 2.0  # Default EPL average
        end
    end
end

# -----------------------------------------------------------------------------
# Build feature matrix from the processed DataFrame for MLJ
# Returns (X::DataFrame, y::CategoricalArray)
# -----------------------------------------------------------------------------
function build_feature_matrix(df::DataFrame)
    X = select(df, FEATURE_COLS)
    # Coerce all Int columns to Float64 for XGBoost compatibility
    for col in names(X)
        if eltype(X[!, col]) <: Integer
            X[!, col] = Float64.(X[!, col])
        end
    end
    y = df.Result
    return X, y
end

# -----------------------------------------------------------------------------
# Get aggregated stats for a single team (used by the prediction endpoint)
# Returns a NamedTuple of key stats
# -----------------------------------------------------------------------------
function get_team_stats(df::DataFrame, team::String, current_elo::Dict)
    home_rows = df[df.HomeTeam .== team, :]
    away_rows = df[df.AwayTeam .== team, :]

    # Last 5 home games
    last_home = nrow(home_rows) >= 5 ? last(home_rows, 5) : home_rows
    last_away = nrow(away_rows) >= 5 ? last(away_rows, 5) : away_rows

    avg_home_sot = nrow(last_home) > 0 ? mean(last_home.HomeShotsOnTarget) : 5.0
    avg_away_sot = nrow(last_away) > 0 ? mean(last_away.AwayShotsOnTarget) : 4.0
    avg_home_corners = nrow(last_home) > 0 ? mean(last_home.HomeCorners) : 5.0
    avg_away_corners = nrow(last_away) > 0 ? mean(last_away.AwayCorners) : 4.5
    avg_home_ht_goals = nrow(last_home) > 0 ? mean(last_home.HalfTimeHomeGoals) : 0.65
    avg_away_ht_goals = nrow(last_away) > 0 ? mean(last_away.HalfTimeAwayGoals) : 0.45

    # Rolling recent goals from the full dataset
    home_recent_goals = nrow(home_rows) > 0 ? mean(last(home_rows.HomeRecentGoals, min(5, nrow(home_rows)))) : 1.35
    away_recent_goals = nrow(away_rows) > 0 ? mean(last(away_rows.AwayRecentGoals, min(5, nrow(away_rows)))) : 1.10

    elo = get(current_elo, team, 1500.0)

    return (
        elo = elo,
        avg_home_sot = avg_home_sot,
        avg_away_sot = avg_away_sot,
        avg_home_corners = avg_home_corners,
        avg_away_corners = avg_away_corners,
        avg_home_ht_goals = avg_home_ht_goals,
        avg_away_ht_goals = avg_away_ht_goals,
        home_recent_goals = home_recent_goals,
        away_recent_goals = away_recent_goals,
        total_home_games = nrow(home_rows),
        total_away_games = nrow(away_rows),
        home_win_rate = nrow(home_rows) > 0 ? mean(home_rows.FullTimeResult .== "H") : 0.45,
        away_win_rate = nrow(away_rows) > 0 ? mean(away_rows.FullTimeResult .== "A") : 0.28,
        draw_rate = (nrow(home_rows) + nrow(away_rows)) > 0 ?
            (sum(home_rows.FullTimeResult .== "D") + sum(away_rows.FullTimeResult .== "D")) /
            (nrow(home_rows) + nrow(away_rows)) : 0.27,
        avg_goals_scored_home = nrow(home_rows) > 0 ? mean(home_rows.FullTimeHomeGoals) : 1.35,
        avg_goals_scored_away = nrow(away_rows) > 0 ? mean(away_rows.FullTimeAwayGoals) : 1.10,
        avg_goals_conceded_home = nrow(home_rows) > 0 ? mean(home_rows.FullTimeAwayGoals) : 1.20,
        avg_goals_conceded_away = nrow(away_rows) > 0 ? mean(away_rows.FullTimeHomeGoals) : 1.50,
    )
end

# Get stats for all current season teams
function get_all_team_stats(df::DataFrame, current_elo::Dict)
    teams = sort(unique(vcat(df.HomeTeam, df.AwayTeam)))
    return Dict(team => get_team_stats(df, team, current_elo) for team in teams)
end

end # module Features
