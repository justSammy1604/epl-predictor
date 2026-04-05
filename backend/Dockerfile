# =============================================================================
# Dockerfile — EPL Prediction Engine Julia Backend (Memory-Optimized)
# =============================================================================

FROM julia:1.10-slim

WORKDIR /app

# Minimal system dependencies for slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libomp-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only Project.toml
COPY Project.toml ./

# Update registry, then instantiate (no precompile, no resolve)
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

# Start server with memory optimization flags
# --heap-size-hint limits Julia's memory footprint
# -e runs startup.jl before server.jl to pre-warm the compiler
CMD ["julia", "--threads=auto", "--project=.", "-e", "include(\"server.jl\")"]
