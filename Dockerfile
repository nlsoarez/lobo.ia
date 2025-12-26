# Lobo IA - Dockerfile para Railway
# Multi-stage build para otimização de tamanho

FROM python:3.11-slim as builder

# Instala dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia requirements primeiro para cache de layers
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage final
FROM python:3.11-slim

# Instala runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash lobo

WORKDIR /app

# Copia dependências instaladas do builder
COPY --from=builder /root/.local /home/lobo/.local

# Copia código da aplicação
COPY --chown=lobo:lobo . .

# Define PATH para incluir pacotes do usuário
ENV PATH=/home/lobo/.local/bin:$PATH

# Variáveis de ambiente padrão
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    EXECUTION_MODE=simulation \
    LOG_LEVEL=INFO

# Cria diretórios necessários
RUN mkdir -p /app/logs /app/data && \
    chown -R lobo:lobo /app

# Muda para usuário não-root
USER lobo

# Expõe porta do Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Comando padrão (pode ser sobrescrito no Railway)
CMD ["python", "start.py"]
