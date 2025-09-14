CTX?=mypi
C=docker --context $(CTX) compose -f docker-compose.yml

.PHONY: deploy pull up down ps logs restart

deploy: pull up
pull:   ; $(C) pull
up:     ; $(C) up -d
down:   ; $(C) down
ps:     ; $(C) ps
logs:   ; $(C) logs -f
restart:; $(C) restart