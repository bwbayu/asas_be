FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y curl nginx && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD [ "gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app", "--access-logfile", "-", "--timeout", "120"]