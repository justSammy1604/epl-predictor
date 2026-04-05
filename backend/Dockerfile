# =============================================================================
# Dockerfile — EPL Prediction Engine Julia Backend (Simplified & Robust)
# =============================================================================

FROM julia:1.10-bullseye

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only Project.toml first (Manifest will be generated in Docker)
COPY Project.toml ./

# Resolve dependencies explicitly and generate Manifest.toml in Docker
# This ensures Manifest matches the Docker's Julia 1.10 version
RUN julia --project=. -e "using Pkg; Pkg.resolve()" && \
    echo "✓ Dependencies resolved"

# Instantiate WITHOUT auto-precompile (packages compile JIT at startup)
RUN julia --project=. -e "using Pkg; Pkg.instantiate(allow_autoprecomp=false)" && \
    echo "✓ Packages downloaded — ready to start"

# Copy source code and data
COPY *.jl ./
COPY data/ ./data/

EXPOSE 8080

# Health check with generous startup window for JIT compilation
HEALTHCHECK --interval=30s --timeout=30s --start-period=900s --retries=3 \
    CMD curl -sf http://localhost:8080/api/health >/dev/null 2>&1 || exit 1

# Start server with multi-threading
CMD ["julia", "--threads=auto", "--project=.", "server.jl"]
