from flask import Flask, request, jsonify
from flask_cors import CORS
from inference.direct.main_deploy import get_score_direct
from inference.similarity.main_deploy import get_score_similarity
import json, pathlib
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app)

# rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour", "10 per minute"]
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
    return "Hello bayu"

@app.get('/questions')
def get_questions():
    return jsonify(PROMPTS.get('specific-prompt', []))

@app.get('/student_answer')
def get_student_answer():
    dataset_id = request.args.get('dataset_id')
    scenario_data = ANSWERS.get('specific-prompt', [])
    first_entry = scenario_data[0] if scenario_data else {}
    return jsonify(first_entry.get(dataset_id, []))

@app.post('/score')
def predict():
    data = request.get_json()
    answer = data.get('answer', '')
    reference = data.get('reference', '')
    
    # validate empty string
    if not all([validate_input(answer), validate_input(reference)]):
        return jsonify({'error': 'Missing input'}), 400
    
    if(answer == '' or reference == ''):
        return jsonify({'error': 'Missing input'}), 400
    
    # get score
    direct_score = get_score_direct(answer, reference)
    similarity_score = get_score_similarity(answer, reference)

    return jsonify({'direct_score': round(float(direct_score), 2), 'similarity_score': round(float(similarity_score), 2)})

# if __name__ == '__main__':
#     app.run(debug=True)