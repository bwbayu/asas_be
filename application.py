from flask import Flask, request, jsonify
from flask_cors import CORS
from inference.direct.main_deploy import get_score_direct
from inference.similarity.main_deploy import get_score_similarity
import json, pathlib

application = Flask(__name__)
CORS(application)

DATA_PATH = pathlib.Path(__file__).parent / 'data' / 'prompt.json'
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    PROMPTS = json.load(f)

ANSWER_PATH = pathlib.Path(__file__).parent / 'data' / 'answer.json'
with open(ANSWER_PATH, 'r', encoding='utf-8') as f:
    ANSWERS = json.load(f)

def validate_input(text: str) -> bool:
    return bool(text and text.strip())

@application.get('/questions')
def get_questions():
    scenario = request.args.get('scenario')
    return jsonify(PROMPTS.get(scenario, []))

@application.get('/student_answer')
def get_student_answer():
    scenario = request.args.get('scenario')
    dataset_id = request.args.get('dataset_id')
    scenario_data = ANSWERS.get(scenario, [])
    first_entry = scenario_data[0] if scenario_data else {}
    return jsonify(first_entry.get(dataset_id, []))

@application.post('/score')
def predict():
    data = request.get_json()
    answer = data.get('answer', '')
    reference = data.get('reference', '')
    scenario = data.get('scenario', '')

    # validate empty string
    if not all([validate_input(answer), validate_input(reference), validate_input(scenario)]):
        return jsonify({'error': 'Missing input'}), 400
    
    # get score
    direct_score = get_score_direct(answer, reference, scenario)
    similarity_score = get_score_similarity(answer, reference, scenario)

    return jsonify({'direct_score': round(float(direct_score), 2), 'similarity_score': round(float(similarity_score), 2)})

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=True)