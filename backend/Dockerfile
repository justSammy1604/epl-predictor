# =============================================================================
# Dockerfile — EPL Prediction Engine Julia Backend
# Strategy: Instantiate only (no resolve/precompile in Docker)
#           Manifest should be pre-generated and committed with Project.toml
# =============================================================================

FROM julia:1.10-bullseye

WORKDIR /app

# Install system dependencies (XGBoost, OpenMP)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libomp-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Project.toml and Manifest.toml
# Both files must exist and be in sync for reliable Docker builds
COPY Project.toml Manifest.toml* ./

# Instantiate ONLY — no resolve/precompile
# If Manifest.toml exists, this just downloads packages
# If only Project.toml exists, Julia will try to resolve (avoid this)
RUN julia --project=. -e " \
    using Pkg; \
    Pkg.instantiate(verbose=true); \
    println(\"Packages instantiated successfully\") \
" || { echo 'ERROR: Package instantiation failed. Ensure Manifest.toml is committed.'; exit 1; }

# Copy source code
COPY *.jl ./

# Copy dataset
COPY data/ ./data/

EXPOSE 8080

# Extended health check for JIT compilation on first boot
HEALTHCHECK --interval=30s --timeout=20s --start-period=900s --retries=5 \
    CMD curl -sf http://localhost:8080/api/health > /dev/null || exit 1

# Run with multiple threads for Monte Carlo performance
CMD ["julia", "--threads=auto", "--project=.", "server.jl"]
