FROM n8nio/n8n:2.3.0

# Passer root pour installer Python
USER root

# Installer python3 et pip sur Alpine
RUN apk add --no-cache python3 py3-pip

# Créer un venv pour les libs Python
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier requirements + installer les dépendances
COPY src/requirements.txt /scripts/requirements.txt
RUN pip3 install --no-cache-dir -r /scripts/requirements.txt

# Copier les scripts
COPY src/fetcher.py /scripts/fetcher.py
COPY src/prices.py /scripts/prices.py
COPY src/analytics.py /scripts/analytics.py
COPY src/main.py /scripts/main.py
COPY src/portfolio.py /scripts/portfolio.py

# Remettre l'utilisateur node (comme dans l'image officielle n8n)
USER node