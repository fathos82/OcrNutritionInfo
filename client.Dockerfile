# ============ Estágio de Build ============
FROM python:3.10-slim as builder

# Instala apenas as dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ============ Estágio Final ============ 
FROM python:3.10-slim

# Copia apenas os pacotes do sistema do builder
COPY --from=builder /usr/share/tesseract-ocr /usr/share/tesseract-ocr
COPY --from=builder /usr/lib/x86_64-linux-gnu/{libtesseract.so.4,liblept.so.5} /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/bin/{tesseract,ffmpeg} /usr/bin/

# Configuração do ambiente
WORKDIR /usr/src/app
ENV PYTHONUNBUFFERED=1

# Instala dependências Python diretamente (sem estágio separado)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY . .

ENTRYPOINT ["sh", "-c", "python main.py \
    --pipeline \"${PIPELINE_ARG:-deploy}\" \
    --runner client \
    --processing-name \"${NAME_ARG:-default-process}\""]