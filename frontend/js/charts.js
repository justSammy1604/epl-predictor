// =============================================================================
// FILE: frontend/js/charts.js
// PURPOSE: All Chart.js visualizations (probability bars, EV charts,
//          Monte Carlo standings, CI error bars)
// REQUIRES: Chart.js loaded in index.html
// =============================================================================

// Colour palette
const COLOURS = {
  home:    "#3b82f6",   // blue
  draw:    "#f59e0b",   // amber
  away:    "#ef4444",   // red
  green:   "#22c55e",
  muted:   "#6b7280",
  bg:      "#1e293b",
  surface: "#0f172a",
  title:   "#22c55e",
  championsLeague: "#3b82f6",
  europaLeague:    "#8b5cf6",
  relegation:      "#ef4444",
};

let predChart = null;
let evChart   = null;
let mcChart   = null;

// =============================================================================
// PREDICTION CHART — Win probability bars with CI error bars
// =============================================================================
export function renderPredictionChart(data) {
  const ctx = document.getElementById("predictionChart").getContext("2d");
  if (predChart) predChart.destroy();

  const labels  = [`${data.home_team} Win`, "Draw", `${data.away_team} Win`];
  const probs   = [data.home_win, data.draw, data.away_win].map(v => +(v * 100).toFixed(1));
  const colours = [COLOURS.home, COLOURS.draw, COLOURS.away];

  // CI error bars (if available)
  let errorBars = null;
  if (data.confidence_intervals) {
    const ci = data.confidence_intervals;
    errorBars = [
      { plus: (ci.home_win.high - data.home_win) * 100, minus: (data.home_win - ci.home_win.low) * 100 },
      { plus: (ci.draw.high    - data.draw)     * 100, minus: (data.draw    - ci.draw.low)     * 100 },
      { plus: (ci.away_win.high - data.away_win) * 100, minus: (data.away_win - ci.away_win.low) * 100 },
    ];
  }

  predChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: probs,
        backgroundColor: colours.map(c => c + "cc"),
        borderColor:     colours,
        borderWidth: 2,
        borderRadius: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const v = ctx.parsed.y.toFixed(1);
              if (!errorBars) return `${v}%`;
              const eb = errorBars[ctx.dataIndex];
              return [`${v}%`, `95% CI: ±${Math.max(eb.plus, eb.minus).toFixed(1)}%`];
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { color: "#94a3b8", callback: v => `${v}%` },
          grid:  { color: "#1e293b" }
        },
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } }
      },
      animation: { duration: 600, easing: "easeInOutQuart" }
    },
    plugins: [{
      // Custom CI error bar plugin
      id: "errorBars",
      afterDraw(chart) {
        if (!errorBars) return;
        const { ctx, scales: { x, y } } = chart;
        ctx.save();
        ctx.strokeStyle = "#f8fafc";
        ctx.lineWidth   = 2;
        chart.data.datasets[0].data.forEach((val, i) => {
          const eb   = errorBars[i];
          const barX = x.getPixelForValue(i);
          const yTop = y.getPixelForValue(val + eb.plus);
          const yBot = y.getPixelForValue(val - eb.minus);
          const hw   = 6;
          ctx.beginPath();
          ctx.moveTo(barX, yTop);  ctx.lineTo(barX, yBot);
          ctx.moveTo(barX - hw, yTop); ctx.lineTo(barX + hw, yTop);
          ctx.moveTo(barX - hw, yBot); ctx.lineTo(barX + hw, yBot);
          ctx.stroke();
        });
        ctx.restore();
      }
    }]
  });
}

// =============================================================================
// BETTING EV CHART — Expected value comparison per outcome
// =============================================================================
export function renderEVChart(data) {
  const ctx = document.getElementById("evChart").getContext("2d");
  if (evChart) evChart.destroy();

  const labels = data.analyses.map(a => a.outcome);
  const evVals = data.analyses.map(a => +(a.expected_value * 100).toFixed(2));
  const barColours = evVals.map(v => v > 0 ? COLOURS.green + "cc" : COLOURS.away + "cc");
  const borderColours = evVals.map(v => v > 0 ? COLOURS.green : COLOURS.away);

  evChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Expected Value (%)",
          data: evVals,
          backgroundColor: barColours,
          borderColor:     borderColours,
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: "Model Prob",
          data: data.analyses.map(a => +(a.model_prob * 100).toFixed(1)),
          type: "line",
          borderColor: COLOURS.home,
          backgroundColor: "transparent",
          pointBackgroundColor: COLOURS.home,
          pointRadius: 5,
          tension: 0.3,
          yAxisID: "prob",
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } },
        tooltip: {
          callbacks: {
            label: ctx => ctx.datasetIndex === 0
              ? `EV: ${ctx.parsed.y.toFixed(2)}%`
              : `Model: ${ctx.parsed.y.toFixed(1)}%`
          }
        }
      },
      scales: {
        y: {
          ticks:  { color: "#94a3b8", callback: v => `${v}%` },
          grid:   { color: "#1e293b" }
        },
        prob: {
          position: "right",
          min: 0, max: 100,
          ticks:  { color: COLOURS.home, callback: v => `${v}%` },
          grid:   { display: false }
        },
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } }
      }
    }
  });
}

// =============================================================================
// MONTE CARLO CHART — Horizontal bar chart of title / top4 / relegation %
// =============================================================================
export function renderMonteCarloChart(data) {
  const ctx = document.getElementById("mcChart").getContext("2d");
  if (mcChart) mcChart.destroy();

  const standings = data.standings.slice(0, 20);
  const labels    = standings.map(r => r.team);
  const titleProb = standings.map(r => r.prob_title);
  const top4Prob  = standings.map(r => r.prob_top4);
  const relProb   = standings.map(r => r.prob_relegation);

  mcChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Title %",
          data: titleProb,
          backgroundColor: COLOURS.title + "cc",
          borderColor:     COLOURS.title,
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: "Top 4 %",
          data: top4Prob,
          backgroundColor: COLOURS.championsLeague + "88",
          borderColor:     COLOURS.championsLeague,
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: "Relegation %",
          data: relProb,
          backgroundColor: COLOURS.relegation + "88",
          borderColor:     COLOURS.relegation,
          borderWidth: 1,
          borderRadius: 4,
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#94a3b8" } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)}%` } }
      },
      scales: {
        x: {
          ticks:  { color: "#94a3b8", callback: v => `${v}%` },
          grid:   { color: "#1e293b" }
        },
        y: {
          ticks:  { color: "#e2e8f0", font: { size: 11 } },
          grid:   { display: false }
        }
      }
    }
  });
}
