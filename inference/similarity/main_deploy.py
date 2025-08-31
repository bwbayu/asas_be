from .modeling_similarity import SimilaritySpecific
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
import warnings
import joblib
import regex as re
import os
from google.cloud import storage

warnings.simplefilter("ignore")
MODEL_NAME = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SimilaritySpecific(MODEL_NAME)

# download model from s3
GCS_BUCKET = os.getenv("GCS_BUCKET", "model-asas-demo")
MODEL_DIR = "/tmp/model_similarity"
GCS_BLOB_PT  = "model_similarity/model_0.pt"
GCS_BLOB_PKL = "model_similarity/reg_0.pkl"

def download_model_from_gcs(blob_name: str, local_filename: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, local_filename)

    if not os.path.exists(local_path):
        client = storage.Client() 
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        print(f"Downloaded gs://{GCS_BUCKET}/{blob_name} -> {local_path}")

    return local_path

model_specific_path = download_model_from_gcs(GCS_BLOB_PT,  "model_0.pt")
pkl_specific_path   = download_model_from_gcs(GCS_BLOB_PKL, "reg_0.pkl")

checkpoint_specific = torch.load(model_specific_path,  map_location=torch.device('cpu'))
reg_model_specific = joblib.load(pkl_specific_path)
model.load_state_dict(checkpoint_specific)
model.eval()

def preprocess_text(text):
    text = text.lower()  # Ubah ke lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text)  # Hapus karakter khusus
    text = ' '.join(text.split())  # Hapus spasi berlebih
    return text

def get_score_similarity(answer: str, reference: str):
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
        reg_model = reg_model_specific

        reference_emb = model(**encoding_reference)
        answer_emb = model(**encoding_answer)

        ref_embedding = F.normalize(reference_emb, p=2, dim=1)
        ans_embedding = F.normalize(answer_emb, p=2, dim=1)

        similarity = F.cosine_similarity(ref_embedding, ans_embedding, dim=1)
        similarity = torch.clamp(similarity, -1.0, 1.0)
        score = reg_model.predict(similarity.reshape(-1, 1))

        return round(score[0], 2)
    
    return 0
