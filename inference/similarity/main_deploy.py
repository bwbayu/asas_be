from .modeling_similarity_specific import SimilaritySpecific
from .modeling_similarity_cross import SimilarityCross
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
import warnings
import joblib
import regex as re
import os
import boto3

warnings.simplefilter("ignore")
MODEL_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_specific = SimilaritySpecific(MODEL_NAME)
model_cross = SimilarityCross(MODEL_NAME)

# download model from s3
S3_BUCKET = "model-asas-bucket"
MODEL_DIR = "model_sim"

def download_model_from_s3(model_key: str, local_filename: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, local_filename)

    if not os.path.exists(local_path):
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, model_key, local_path)
        print(f"Downloaded {model_key} from S3.")

    return local_path

model_specific_path = download_model_from_s3("model_sim/model_0.pt", "model_0.pt")
pkl_specific_path = download_model_from_s3("model_sim/reg_0.pkl", "reg_0.pkl")
model_cross_path = download_model_from_s3("model_sim/model_1.pt", "model_1.pt")
pkl_cross_path = download_model_from_s3("model_sim/reg_1.pkl", "reg_1.pkl")

checkpoint_specific = torch.load(model_specific_path, map_location='cpu')
reg_model_specific = joblib.load(pkl_specific_path)
model_specific.load_state_dict(checkpoint_specific)
model_specific.eval()

checkpoint_cross = torch.load(model_cross_path, map_location='cpu')
reg_model_cross = joblib.load(pkl_cross_path)
model_cross.load_state_dict(checkpoint_cross)
model_cross.eval()

def preprocess_text(text):
    text = text.lower()  # Ubah ke lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text)  # Hapus karakter khusus
    text = ' '.join(text.split())  # Hapus spasi berlebih
    return text

registry = {
    "specific-prompt": {
        "model": model_specific,
        "reg_model": reg_model_specific
    },
    "cross-prompt": {
        "model": model_cross,
        "reg_model": reg_model_cross
    }
}


def get_score_similarity(answer: str, reference: str, scenario: str):
    encoding_reference = tokenizer.encode_plus(
        preprocess_text(reference),
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    encoding_answer = tokenizer.encode_plus(
        preprocess_text(answer),
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        model = registry[scenario]["model"]
        reg_model = registry[scenario]["reg_model"]

        reference_emb = model(**encoding_reference)
        answer_emb = model(**encoding_answer)

        ref_embedding = F.normalize(reference_emb, p=2, dim=1)
        ans_embedding = F.normalize(answer_emb, p=2, dim=1)

        similarity = F.cosine_similarity(ref_embedding, ans_embedding, dim=1)
        similarity = torch.clamp(similarity, -1.0, 1.0)
        score = reg_model.predict(similarity.reshape(-1, 1))

        return round(score[0], 2)
    
    return 0

# print(get_score_similarity("Buah mengandung banyak vitamin seperti vitamin C yang bisa meningkatkan sistem imun, jadi tubuh tidak mudah sakit.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_similarity("Buah itu penting karena bisa bikin kenyang dan segar, jadi kita nggak perlu makan makanan berat.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_similarity("Makan buah bisa bikin tubuh jadi kurus karena buah mengandung banyak lemak yang dibakar saat tidur.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
