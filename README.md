# Project Overview

Flask backend for a demo website that **scores student answers** using two approaches: **direct** and **similarity**. The API provides:

* `GET /questions` → load questions from `data/prompt.json`
* `GET /student_answer` → load sample student answers from `data/answer.json`
* `POST /score` → returns **two scores**: `direct_score` and `similarity_score` (the backend computes both; there’s no `approach` parameter)

- Frontend code : https://github.com/bwbayu/asas_fe

---

## Features

* 3 simple endpoints (questions, student\_answer, score)
* Two scoring pipelines:

  * **direct** → predict score directly from text (`model.pt`)
  * **similarity** → score via similarity to a reference (`model.pt` + Linear Regression `.pkl`)
* **Local** mode (load models from disk) and **Deploy** mode (download models from **GCS**)
* Rate limiting via Flask-Limiter: `200/day`, `100/hour`, `20/minute`
* Runs via **Python 3.10** or **Docker** (Gunicorn inside the container on port 8000)

---

## Project Structure

```
.
├─ app.py                             # Flask app: /questions, /student_answer, /score
├─ .env.example                       # Environment variables (GCS/AWS S3)
├─ Dockerfile
├─ docker-compose.yaml
├─ requirements.txt
├─ data/
│  ├─ prompt.json                     # Questions
│  └─ answer.json                     # Sample student answers
└─ inference/
   ├─ direct/
   │  ├─ main.py                      # Local mode (no download)
   │  ├─ main_deploy.py               # Deploy mode (downloads from GCS)
   │  └─ modelling_direct.py          # Direct approach model code
   └─ similarity/
      ├─ main.py                      # Local mode
      ├─ main_deploy.py               # Deploy mode (downloads from GCS)
      └─ modelling_similarity.py      # Similarity approach model code (Linear Regression)
```

**Model locations on GCS:**

* Direct: `model_direct/model.pt`
* Similarity: `model_similarity/model.pt` **and** `model_similarity/model.pkl`

**Local model locations (if used):**

* `inference/direct/model/model.pt`
* `inference/similarity/model/model.pt`
* `inference/similarity/model/model.pkl`

---

## Requirements

* **Python 3.10**
* `pip` / `venv`
* (optional) **Docker** & **Docker Compose**
* GCS access (for deploy mode)

---

## Environment (GCS)

Copy `.env.example` → `.env` and fill:

```
# GCS
GCS_BUCKET=your-bucket-name
```

---

## Run Modes (important)

* **Local (models on disk)**
  In `app.py`, import:

  * `from inference.direct.main import ...`
  * `from inference.similarity.main import ...`

* **Deploy (models on GCS)**
  In `app.py`, import:

  * `from inference.direct.main_deploy import ...`
  * `from inference.similarity.main_deploy import ...`

If you mess up imports, the app will either fail to load models or try to download them when you intended local.

---

## How to Run

### Python

```bash
# 1) (optional) create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Ensure imports in app.py match your mode (local/deploy)

# 4) Run (uncomment app.run in app.py if you previously disabled it)
python app.py

# 5) Access
# http://localhost:5000
```

### Docker

**Container listens on port `8000` (Gunicorn). Map any host port you want to `8000`.**

1. **Build the image**

```bash
docker build -t <image-name> .
# e.g.
docker build -t asas-be .
```

2. **Run the container**

```bash
docker run -d --name <container-name> -p <host_port>:8000 --env-file .env <image-name>
# e.g. (host 5001 → container 8000)
docker run -d --name asas-be -p 5001:8000 --env-file .env asas-be
```

> Using GCS with a Service Account key? Mount it:

```bash
docker run -d --name asas-be \
  -p 5001:8000 \
  --env-file .env \
  -v $PWD/creds.json:/app/creds.json:ro \
  asas-be
```

3. **Access the API**

```
http://localhost:<host_port>
# e.g. http://localhost:5001
```

4. **Stop the container**

```bash
docker stop <container-name>
# e.g.
docker stop asas-be
```

5. **Remove the container**

```bash
docker rm <container-name>
# e.g.
docker rm asas-be
```

6. **Remove the image** (optional)

```bash
docker image rm <image-name>
# e.g.
docker image rm asas-be
```


### Docker Compose

```bash
docker compose up --build -d
```

---

## API Reference

### `GET /questions`

Returns questions from `data/prompt.json`.

**200 Example**

```json
{
  "data": [
    {
        "question":"question 1",
        "reference_answer":"reference answer 1",
        "dataset_id":1
    },
  ],
  "count": 1
}
```

### `GET /student_answer`

Returns sample student answers from `data/answer.json`.

**200 Example**

```json
{
  "data": [
    {"1": [
        {
            "answer": "answer 1",
            "score": 0.21
        },
    ]}
  ],
  "count": 1
}
```

### `POST /score`

Computes **two scores**: `direct_score` and `similarity_score`.
There is **no** `approach` parameter — the backend runs both pipelines.

**Request Example**

```json
{
  "answer": "This is my answer...",
  "reference": "Reference solution or rubric..."
}
```

**200 Example**

```json
{
  "direct_score": 0.83,
  "similarity_score": 0.76
}
```

**400 Example**

```json
{ "error": "Missing input" }
```

**429 Example (rate limit)**

```json
{ "error": "Too many requests. Please retry later." }
```

---

## Rate Limiting

Default:

```
default_limits = ["200 per day", "100 per hour", "20 per minute"]
```
