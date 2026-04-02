# =============================================================================
# FILE: monte_carlo.jl
# PURPOSE: Monte Carlo Season Simulator
#          Runs N simulations of the remaining EPL season using model
#          probabilities. Outputs expected standings with confidence intervals.
# USED BY: server.jl (POST /api/monte-carlo)
# =============================================================================

module MonteCarlo

using DataFrames, Statistics, Random, StatsBase

export simulate_season, SimulationResult

# Current active EPL clubs (can be updated each season)
const CURRENT_EPL_TEAMS = [
    "Arsenal", "Aston Villa", "Brentford", "Brighton", "Burnley",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
    "Luton", "Man City", "Man United", "Newcastle", "Nott'm Forest",
    "Sheffield United", "Southampton", "Tottenham", "West Ham", "Wolves"
]

struct SimulationResult
    team::String
    mean_points::Float64
    std_points::Float64
    ci_low::Float64           # 5th percentile
    ci_high::Float64          # 95th percentile
    mean_position::Float64
    prob_top4::Float64        # Champions League spots
    prob_top6::Float64        # Europa League
    prob_relegation::Float64  # Bottom 3
    prob_title::Float64       # League winner
end

# -----------------------------------------------------------------------------
# generate_fixture_list
# Creates a round-robin fixture list for a list of teams
# -----------------------------------------------------------------------------
function generate_fixture_list(teams::Vector{String})
    fixtures = Tuple{String,String}[]
    for i in 1:length(teams)
        for j in 1:length(teams)
            i != j && push!(fixtures, (teams[i], teams[j]))
        end
    end
    return fixtures
end

# -----------------------------------------------------------------------------
# simulate_season
# Input:
#   predict_fn — function(home::String, away::String) → (home_win, draw, away_win)
#   played     — Dict mapping (home, away) → actual result "H"/"D"/"A"
#   teams      — list of teams in the league
#   n_sims     — number of Monte Carlo iterations
# Output:
#   Vector{SimulationResult} sorted by mean_points descending
# -----------------------------------------------------------------------------
function simulate_season(
    predict_fn::Function,
    played::Dict = Dict();
    teams::Vector{String} = CURRENT_EPL_TEAMS,
    n_sims::Int = 5000
)
    fixtures = generate_fixture_list(teams)
    n_teams  = length(teams)

    # Pre-compute model probabilities for unplayed fixtures
    @info "Pre-computing probabilities for $(length(fixtures)) fixtures..."
    prob_cache = Dict{Tuple{String,String}, Tuple{Float64,Float64,Float64}}()

    for (home, away) in fixtures
        key = (home, away)
        if !haskey(played, key)
            try
                p = predict_fn(home, away)
                prob_cache[key] = (p.home_win, p.draw, p.away_win)
            catch
                # Fallback to historical EPL averages if team not in training data
                prob_cache[key] = (0.46, 0.27, 0.27)
            end
        end
    end

    # Points ledger for each simulation — shape: (n_sims × n_teams)
    team_idx = Dict(t => i for (i, t) in enumerate(teams))
    all_points = zeros(Int, n_sims, n_teams)

    # Seed already-played results (these are the same in every simulation)
    base_points = zeros(Int, n_teams)
    for ((home, away), result) in played
        !haskey(team_idx, home) && continue
        !haskey(team_idx, away) && continue
        hi = team_idx[home]
        ai = team_idx[away]
        if result == "H"
            base_points[hi] += 3
        elseif result == "D"
            base_points[hi] += 1
            base_points[ai] += 1
        else
            base_points[ai] += 3
        end
    end

    @info "Running $n_sims Monte Carlo simulations..."
    Threads.@threads for sim in 1:n_sims
        pts = copy(base_points)

        for (home, away) in fixtures
            haskey(played, (home, away)) && continue
            !haskey(team_idx, home) && continue
            !haskey(team_idx, away) && continue

            ph, pd, pa = prob_cache[(home, away)]

            # Sample outcome from multinomial distribution
            r = rand()
            hi = team_idx[home]
            ai = team_idx[away]
            if r < ph
                pts[hi] += 3
            elseif r < ph + pd
                pts[hi] += 1
                pts[ai] += 1
            else
                pts[ai] += 3
            end
        end

        all_points[sim, :] .= pts
    end

    @info "Aggregating simulation results..."

    # Per-team statistics across all simulations
    results = SimulationResult[]

    for (i, team) in enumerate(teams)
        pts_vec = all_points[:, i]

        # Rank team in each simulation (1 = winner)
        positions = zeros(Int, n_sims)
        for sim in 1:n_sims
            row = all_points[sim, :]
            # Rank = 1 + number of teams with strictly more points
            positions[sim] = 1 + sum(row .> pts_vec[sim])
        end

        push!(results, SimulationResult(
            team,
            mean(pts_vec),
            std(pts_vec),
            quantile(pts_vec, 0.05),
            quantile(pts_vec, 0.95),
            mean(positions),
            mean(positions .<= 4),
            mean(positions .<= 6),
            mean(positions .>= (n_teams - 2)),
            mean(positions .== 1)
        ))
    end

    # Sort by mean points descending
    sort!(results, by = r -> r.mean_points, rev = true)
    return results
end

end # module MonteCarlo
