FROM n8nio/n8n:latest

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

# Copier ton script
COPY src/price_data.py /scripts/price_data.py
COPY src/portfolio.py /scripts/portfolio.py

# Remettre l'utilisateur node (comme dans l'image officielle n8n)
USER node