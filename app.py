from flask import Flask, request, jsonify
from flask_cors import CORS
from inference.direct.main_deploy import get_score_direct
from inference.similarity.main_deploy import get_score_similarity
import json, pathlib
from rate_limiter import allow, allow_daily_global_firestore, DAILY_LIMIT
import math

app = Flask(__name__)
CORS(
    app,
    origins=['https://asas-demo.web.app'],
    expose_headers=[
        'Retry-After',
        'X-RateLimit-Limit-Day',
        'X-RateLimit-Remaining-Day',
        'X-RateLimit-Reset-Seconds',
    ],
)

DATA_PATH = pathlib.Path(__file__).parent / 'data' / 'prompt.json'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    PROMPTS = json.load(f)

ANSWER_PATH = pathlib.Path(__file__).parent / 'data' / 'answer.json'
with open(ANSWER_PATH, 'r', encoding='utf-8') as f:
    ANSWERS = json.load(f)

def validate_input(text: str) -> bool:
    return bool(text and text.strip())

@app.get('/')
def main():
    return "Hello"

@app.get('/questions')
def get_questions():
    return jsonify(PROMPTS.get('specific-prompt', []))

@app.get('/student_answer')
def get_student_answer():
    scenario_data = ANSWERS.get('specific-prompt', [])
    first_entry = scenario_data[0] if scenario_data else {}
    return jsonify(first_entry)

@app.post('/score')
async def predict():
    data = request.get_json()
    answer = data.get('answer', '')
    reference = data.get('reference', '')
    
    # validate empty string
    if not all([validate_input(answer), validate_input(reference)]):
        return jsonify({'error': 'Missing input'}), 400
    
    if(answer == '' or reference == ''):
        return jsonify({'error': 'Missing input'}), 400
    
    # global daily rate limit
    allowed_daily, remaining, retry_after = await allow_daily_global_firestore()
    if not allowed_daily:
        resp = jsonify({"detail": "Daily cap reached"})
        resp.status_code = 429
        resp.headers.update({
            "Retry-After": str(int(math.ceil(retry_after or 0))),
            "X-RateLimit-Limit-Day": str(DAILY_LIMIT),
            "X-RateLimit-Remaining-Day": str(remaining),
            "X-RateLimit-Reset-Seconds": str(int(retry_after or 0)),
        })
        return resp

    # rate-limit per IP
    ip = (request.headers.get('X-Forwarded-For', '').split(',')[0].strip()
          or request.remote_addr
          or "unknown")
    # check rate limit
    allowed, retry_after_ip = allow(ip)
    # show error message because rate limit exceed
    if not allowed:
        resp = jsonify({"detail": "Too Many Requests"})
        resp.status_code = 429
        resp.headers.update({
            "Retry-After": str(int(math.ceil(retry_after_ip or 1))),
        })
        return resp
    
    # get score
    direct_score = get_score_direct(answer, reference)
    similarity_score = get_score_similarity(answer, reference)

    return jsonify({
        'direct_score': round(float(direct_score), 2),
        'similarity_score': round(float(similarity_score), 2)
    })

# if __name__ == '__main__':
#     app.run(debug=True)