CTX?=mypi
C=docker --context $(CTX) compose -f docker-compose.yml
IMAGE_NAME = n8n-python:latest

.PHONY: deploy pull up down ps logs restart

build:
	docker --context $(CTX) build -t $(IMAGE_NAME) .

deploy: pull up
pull:   ; $(C) pull
up:     ; $(C) up -d
down:   ; $(C) down
ps:     ; $(C) ps
logs:   ; $(C) logs -f
restart:; $(C) restart