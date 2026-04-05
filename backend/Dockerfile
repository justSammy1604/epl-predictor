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

# Copy only Project.toml
COPY Project.toml ./

# Update registry, then instantiate (no precompile, no resolve)
# Registry.update() populates the package lookup table
# instantiate() automatically resolves from Project.toml
RUN julia --project=. -e " \
    using Pkg; \
    Pkg.Registry.update(); \
    Pkg.instantiate(allow_autoprecomp=false); \
" && echo "✓ Setup complete"

# Copy source code and data
COPY *.jl ./
COPY data/ ./data/

EXPOSE 8080

# Health check with generous startup window for JIT compilation
HEALTHCHECK --interval=30s --timeout=30s --start-period=900s --retries=3 \
    CMD curl -sf http://localhost:8080/api/health >/dev/null 2>&1 || exit 1

# Start server with multi-threading
CMD ["julia", "--threads=auto", "--project=.", "server.jl"]
