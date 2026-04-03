# =============================================================================
# FILE: model.jl
# PURPOSE: XGBoost Model — Training, Hyperparameter Tuning, Prediction,
#          Confidence Intervals via Bootstrap
# USED BY: server.jl
# =============================================================================

module Model

using MLJ, DataFrames, Statistics, Random, CSV, CategoricalArrays
using ..Features

export train_model!, predict_match, predict_with_confidence,
       load_or_train_model, MODEL_PATH

const MODEL_PATH = joinpath(@__DIR__, "data", "footballmodel.jlso")

# Holds trained machine and ELO state — populated at startup
mutable struct ModelState
    mach::Any           # MLJ tuned machine
    current_elo::Dict   # Final ELO ratings after all training data
    df::DataFrame       # Full processed dataset
    teams::Vector{String}
    is_trained::Bool
end

ModelState() = ModelState(nothing, Dict(), DataFrame(), String[], false)

# Global singleton — server.jl initializes this once at startup
const STATE = ModelState()

# -----------------------------------------------------------------------------
# TRAINING PIPELINE
# Mirrors your Pluto notebook exactly:
#  1. Load CSV
#  2. Compute ELO ratings chronologically
#  3. Compute rolling 5-game goal averages
#  4. Encode target as categorical
#  5. 80/20 train/test split
#  6. XGBoostClassifier with Grid-search CV on max_depth ∈ [3,9]
# -----------------------------------------------------------------------------
function train_model!(data_path::String)
    @info "Loading dataset from $data_path..."
    df = CSV.read(data_path, DataFrame)
    df.MatchDate = Date.(df.MatchDate)

    @info "Computing yellow cards statistics ($(nrow(df)) matches)..."
    Features.compute_yellow_cards!(df)

    df.Result = categorical(df.FullTimeResult)

    X, y = Features.build_feature_matrix(df)

    @info "Splitting 80/20 train/test..."
    Random.seed!(42)
    train_idx, test_idx = partition(eachindex(y), 0.8, shuffle=true)
    Xtrain, Xtest = X[train_idx, :], X[test_idx, :]
    ytrain, ytest = y[train_idx], y[test_idx]

    @info "Training XGBoostClassifier with grid-search CV (max_depth 3..9)..."
    XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost verbosity=0

    model = XGBoostClassifier(
        num_round=100,
        eta=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )

    r = range(model, :max_depth, lower=3, upper=9)
    tuned_model = TunedModel(
        model=model,
        resampling=CV(nfolds=5),
        tuning=Grid(resolution=7),
        range=r,
        measure=accuracy
    )

    mach = machine(tuned_model, Xtrain, ytrain)
    MLJ.fit!(mach, verbosity=1)

    # Evaluate on held-out test set
    yhat = MLJ.predict_mode(mach, Xtest)
    acc = accuracy(yhat, ytest)
    @info "Test accuracy: $(round(acc * 100, digits=2))%"

    # Save model
    MLJ.save(MODEL_PATH, mach)
    @info "Model saved to $MODEL_PATH"

    # Populate global state
    STATE.mach = mach
    STATE.df = df
    STATE.teams = sort(unique(vcat(df.HomeTeam, df.AwayTeam)))
    STATE.is_trained = true

    return acc
end

# Try loading saved model; train from scratch if not found
function load_or_train_model(data_path::String)
    if isfile(MODEL_PATH)
        @info "Found saved model — loading $MODEL_PATH..."
        try
            # We still need the processed df for feature lookup
            df = CSV.read(data_path, DataFrame)
            df.MatchDate = Date.(df.MatchDate)
            Features.compute_yellow_cards!(df)
            df.Result = categorical(df.FullTimeResult)

            mach = machine(MODEL_PATH)
            STATE.mach = mach
            STATE.df = df
            STATE.teams = sort(unique(vcat(df.HomeTeam, df.AwayTeam)))
            STATE.is_trained = true
            @info "Model loaded successfully."
            return
        catch e
            @warn "Failed to load saved model: $e — retraining..."
        end
    end
    train_model!(data_path)
end

# -----------------------------------------------------------------------------
# POINT PREDICTION
# Returns raw probabilities for [A, D, H] classes from XGBoost
# Uses the latest historical stats for each team as input features
# -----------------------------------------------------------------------------
function predict_match(home_team::String, away_team::String)
    @assert STATE.is_trained "Model not trained — call load_or_train_model first"
    df = STATE.df

    # Get team stats (contains historical averages)
    home_stats = Features.get_team_stats(df, home_team, Dict())
    away_stats = Features.get_team_stats(df, away_team, Dict())

    # Compute home yellow cards (average from last 5 home games)
    home_games = df[df.HomeTeam .== home_team, :]
    home_yellow = nrow(home_games) > 0 ? 
        mean(last(home_games.HomeYellowCards, min(5, nrow(home_games)))) : 2.0

    # Compute away yellow cards (average from last 5 away games)
    away_games = df[df.AwayTeam .== away_team, :]
    away_yellow = nrow(away_games) > 0 ? 
        mean(last(away_games.AwayYellowCards, min(5, nrow(away_games)))) : 2.0

    # Build input row matching FEATURE_COLS order exactly
    input = DataFrame(
        AwayShotsOnTarget = [away_stats.avg_away_sot],
        HomeShotsOnTarget = [home_stats.avg_home_sot],
        HalfTimeHomeGoals = [home_stats.avg_home_ht_goals],
        HalfTimeAwayGoals = [away_stats.avg_away_ht_goals],
        HomeYellowCards = [home_yellow],
        AwayYellowCards = [away_yellow],
        HomeCorners = [home_stats.avg_home_corners],
        AwayCorners = [away_stats.avg_away_corners]
    )

    probs_dist = MLJ.predict(STATE.mach, input)
    dist = probs_dist[1]

    classes = levels(dist)  # Typically ["A", "D", "H"]
    probs = Dict(c => pdf(dist, c) for c in classes)

    return (
        home_win  = get(probs, "H", 0.0),
        draw      = get(probs, "D", 0.0),
        away_win  = get(probs, "A", 0.0),
    )
end

# -----------------------------------------------------------------------------
# BOOTSTRAP CONFIDENCE INTERVALS
# Runs N bootstrap samples of recent team data to build a distribution
# over predicted probabilities, then returns mean ± 95% CI per outcome
# This gives honest uncertainty quantification around the point estimates
# -----------------------------------------------------------------------------
function predict_with_confidence(
    home_team::String,
    away_team::String;
    n_bootstrap::Int = 200
)
    @assert STATE.is_trained "Model not trained"
    df = STATE.df

    home_rows_all = df[df.HomeTeam .== home_team, :]
    away_rows_all = df[df.AwayTeam .== away_team, :]

    if nrow(home_rows_all) < 5 || nrow(away_rows_all) < 5
        pt = predict_match(home_team, away_team)
        # Widen CIs for small-sample teams
        ci_width = 0.08
        return (
            home_win  = pt.home_win,
            draw      = pt.draw,
            away_win  = pt.away_win,
            home_win_ci = (max(0.0, pt.home_win - ci_width), min(1.0, pt.home_win + ci_width)),
            draw_ci     = (max(0.0, pt.draw - ci_width),     min(1.0, pt.draw + ci_width)),
            away_win_ci = (max(0.0, pt.away_win - ci_width), min(1.0, pt.away_win + ci_width)),
            n_bootstrap = 0,
            note = "Small sample — CI widened heuristically"
        )
    end

    # Bootstrap: resample from each team's recent 20 home/away games
    home_pool = last(home_rows_all, min(20, nrow(home_rows_all)))
    away_pool = last(away_rows_all, min(20, nrow(away_rows_all)))

    home_probs = Float64[]
    draw_probs = Float64[]
    away_probs = Float64[]

    for _ in 1:n_bootstrap
        # Resample with replacement from recent games
        hi = rand(1:nrow(home_pool), min(5, nrow(home_pool)))
        ai = rand(1:nrow(away_pool), min(5, nrow(away_pool)))

        hrows = home_pool[hi, :]
        arows = away_pool[ai, :]

        input = DataFrame(
            AwayShotsOnTarget = [mean(arows.AwayShotsOnTarget)],
            HomeShotsOnTarget = [mean(hrows.HomeShotsOnTarget)],
            HalfTimeHomeGoals = [mean(hrows.HalfTimeHomeGoals)],
            HalfTimeAwayGoals = [mean(arows.HalfTimeAwayGoals)],
            HomeYellowCards   = [mean(hrows.HomeYellowCards)],
            AwayYellowCards   = [mean(arows.AwayYellowCards)],
            HomeCorners       = [mean(hrows.HomeCorners)],
            AwayCorners       = [mean(arows.AwayCorners)]
        )

        dist = MLJ.predict(STATE.mach, input)[1]
        classes = levels(dist)
        p = Dict(c => pdf(dist, c) for c in classes)

        push!(home_probs, get(p, "H", 0.0))
        push!(draw_probs, get(p, "D", 0.0))
        push!(away_probs, get(p, "A", 0.0))
    end

    # 95% CI via percentile bootstrap
    ci(v) = (quantile(v, 0.025), quantile(v, 0.975))

    return (
        home_win  = mean(home_probs),
        draw      = mean(draw_probs),
        away_win  = mean(away_probs),
        home_win_ci = ci(home_probs),
        draw_ci     = ci(draw_probs),
        away_win_ci = ci(away_probs),
        n_bootstrap = n_bootstrap,
        note = "95% bootstrap CI over $n_bootstrap samples"
    )
end

end # module Model
