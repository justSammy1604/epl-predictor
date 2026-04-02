# =============================================================================
# Dockerfile — EPL Prediction Engine Julia Backend
# Used by: Render (Docker environment) and Railway (auto-detected)
# =============================================================================

# Use the official Julia 1.10 image (LTS, stable)
FROM julia:1.10-bullseye

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Julia packages
# (XGBoost and HTTP.jl need these at compile time)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer caching — only re-runs if deps change)
COPY Project.toml ./

# Generate Manifest.toml by resolving + precompiling all packages
# This is the slow step (~5-10 min on first build, cached after that)
RUN julia --project=. -e " \
    using Pkg; \
    Pkg.resolve(); \
    Pkg.instantiate(); \
    Pkg.precompile(); \
    println(\"All packages compiled successfully\") \
"

# Copy source files
COPY *.jl ./

# Copy the dataset
COPY data/ ./data/

# Expose the API port
EXPOSE 8080

# Health check — lets Render/Railway know the container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start the server
# --threads=auto uses all available CPU cores (important for Monte Carlo)
CMD ["julia", "--threads=auto", "--project=.", "server.jl"]
