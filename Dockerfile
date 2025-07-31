FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y curl
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "app:app", "--access-logfile", "-", "--timeout", "120"]