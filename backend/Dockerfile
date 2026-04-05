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

# Copy Project.toml and Manifest.toml (both required)
COPY Project.toml Manifest.toml ./

# Instantiate packages from locked Manifest.toml
# This is fast (2-3 min) and reliable — no resolve/registry calls
RUN julia --project=. -e "using Pkg; Pkg.instantiate()" && \
    echo "✓ Packages instantiated successfully"

# Copy source code and data
COPY *.jl ./
COPY data/ ./data/

EXPOSE 8080

# Health check with generous startup window for JIT compilation
HEALTHCHECK --interval=30s --timeout=30s --start-period=900s --retries=3 \
    CMD curl -sf http://localhost:8080/api/health >/dev/null 2>&1 || exit 1

# Start server with multi-threading
CMD ["julia", "--threads=auto", "--project=.", "server.jl"]
