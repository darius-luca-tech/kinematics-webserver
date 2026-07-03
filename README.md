### Instalare & Rulare

Proiectul poate fi instalat atât folosind codul sursâ, cât si fișierul `docker-compose.yml` pentru a crea ulterior un mediu izolat (container) Docker în care ruleazâ aplicația pe portul 8000:

#### Local

```
git clone https://github.com/darius-luca-tech/kinematics-webserver.git
cd kinematics-webserver

python -m venv .venv
source .venv/bin/activate        # pe Windows: .venv\Scripts\activate

pip install -r requirements.txt

uvicorn server:app --host 0.0.0.0 --port 8000
```

### Docker

```
docker compose up --build
```
