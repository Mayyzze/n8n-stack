FROM n8nio/n8n:1.121.2

USER root

# Install Python via apk (Alpine-based image)
RUN apk add --no-cache python3 py3-pip py3-virtualenv

# Create venv and install dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY src/requirements.txt /scripts/requirements.txt
RUN pip install --no-cache-dir -r /scripts/requirements.txt

# Copy scripts
COPY src/fetcher.py /scripts/fetcher.py
COPY src/prices.py /scripts/prices.py
COPY src/analytics.py /scripts/analytics.py
COPY src/main.py /scripts/main.py
COPY src/portfolio.py /scripts/portfolio.py

USER node
