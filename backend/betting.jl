# =============================================================================
# FILE: betting.jl
# PURPOSE: Betting Expected Value (EV) Calculator + Kelly Criterion Staking
#          Compares our model's implied probability against bookmaker odds
#          to find +EV bets and compute optimal stake size
# USED BY: server.jl (POST /api/betting-ev)
# =============================================================================

module Betting

export BettingAnalysis, compute_betting_ev, kelly_criterion,
       decimal_to_implied_prob, no_vig_probability

struct BettingAnalysis
    outcome::String           # "Home Win" | "Draw" | "Away Win"
    model_prob::Float64       # Our XGBoost probability
    book_odds::Float64        # Decimal odds from bookmaker
    implied_prob::Float64     # Bookmaker's implied probability (raw)
    no_vig_prob::Float64      # Fair implied probability (vig removed)
    expected_value::Float64   # EV per £1 staked = (prob × odds) - 1
    kelly_fraction::Float64   # Optimal bet fraction of bankroll
    is_value_bet::Bool        # True if EV > 0 (model has edge)
    edge::Float64             # Our prob minus book's no-vig prob
end

# -----------------------------------------------------------------------------
# decimal_to_implied_prob
# Convert decimal odds (e.g. 2.50) → raw implied probability (0.40)
# -----------------------------------------------------------------------------
decimal_to_implied_prob(odds::Float64) = 1.0 / odds

# -----------------------------------------------------------------------------
# no_vig_probability
# Removes the bookmaker's margin ("vig") from the three raw implied
# probabilities so they sum to 1.0. This is the "fair" book price.
# -----------------------------------------------------------------------------
function no_vig_probability(home_odds::Float64, draw_odds::Float64, away_odds::Float64)
    raw_home = decimal_to_implied_prob(home_odds)
    raw_draw = decimal_to_implied_prob(draw_odds)
    raw_away = decimal_to_implied_prob(away_odds)

    overround = raw_home + raw_draw + raw_away  # Typically ~1.06 for EPL markets

    return (
        home = raw_home / overround,
        draw = raw_draw / overround,
        away = raw_away / overround,
        vig  = (overround - 1.0) * 100.0  # e.g. 6.0 = 6% margin
    )
end

# -----------------------------------------------------------------------------
# kelly_criterion
# Edward O. Kelly's formula for optimal bet sizing:
#   f* = (bp - q) / b  where b = odds - 1, p = win prob, q = 1 - p
# We apply a half-Kelly fraction (0.5×) as a risk management measure
# Returns fraction of bankroll to stake (clamped to [0, 0.25])
# -----------------------------------------------------------------------------
function kelly_criterion(prob::Float64, decimal_odds::Float64; fraction::Float64 = 0.5)
    b = decimal_odds - 1.0
    q = 1.0 - prob
    full_kelly = (b * prob - q) / b
    half_kelly = fraction * full_kelly
    return clamp(half_kelly, 0.0, 0.25)  # Never stake more than 25% of roll
end

# -----------------------------------------------------------------------------
# compute_betting_ev
# Core function: given our model probabilities + bookmaker odds,
# compute EV, Kelly stake, and whether each outcome is a value bet
# -----------------------------------------------------------------------------
function compute_betting_ev(
    model_home::Float64,
    model_draw::Float64,
    model_away::Float64,
    book_home_odds::Float64,
    book_draw_odds::Float64,
    book_away_odds::Float64
)
    nv = no_vig_probability(book_home_odds, book_draw_odds, book_away_odds)

    outcomes = [
        ("Home Win", model_home, book_home_odds, nv.home),
        ("Draw",     model_draw, book_draw_odds, nv.draw),
        ("Away Win", model_away, book_away_odds, nv.away),
    ]

    analyses = BettingAnalysis[]

    for (label, model_p, book_odds, no_vig_p) in outcomes
        # EV = (probability × payout) - cost
        # On a £1 bet: EV = model_prob × (odds - 1) - (1 - model_prob)
        #                  = model_prob × odds - 1
        ev = model_p * book_odds - 1.0

        kelly = kelly_criterion(model_p, book_odds)
        edge  = model_p - no_vig_p

        push!(analyses, BettingAnalysis(
            label,
            model_p,
            book_odds,
            decimal_to_implied_prob(book_odds),
            no_vig_p,
            ev,
            kelly,
            ev > 0.0,
            edge
        ))
    end

    return (
        analyses = analyses,
        overround = 1.0 / book_home_odds + 1.0 / book_draw_odds + 1.0 / book_away_odds,
        vig_pct   = nv.vig,
        best_bet  = sort(analyses, by = a -> a.expected_value, rev = true)[1],
        has_value = any(a.is_value_bet for a in analyses)
    )
end

end # module Betting
