from .modeling_direct import DirectModel
from transformers import BertTokenizer
import torch
import warnings
import regex as re
import os
from google.cloud import storage

warnings.simplefilter("ignore")
MODEL_NAME = 'indobenchmark/indobert-lite-base-p2'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = DirectModel(MODEL_NAME)

# download model from s3
GCS_BUCKET = os.getenv("GCS_BUCKET", "model-asas-demo")
MODEL_DIR = "/tmp/model_direct"
GCS_BLOB_MODEL = "model_direct/model_1.pt"

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

model_specific_path = download_model_from_gcs(GCS_BLOB_MODEL, "model_1.pt")

# Load weight ke model
checkpoint_specific = torch.load(model_specific_path,  map_location=torch.device('cpu'))
model.load_state_dict(checkpoint_specific)
model.eval()

def preprocess_text(text):
    text = text.lower()  # Ubah ke lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text)  # Hapus karakter khusus
    text = ' '.join(text.split())  # Hapus spasi berlebih
    return text

def get_score_direct(answer: str, reference: str):
    inputs = tokenizer.encode_plus(
            preprocess_text(reference),
            preprocess_text(answer),
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    with torch.no_grad():
        predictions = model(**inputs).squeeze(1)
        score = torch.clamp(predictions, 0, 1)
        return round(score.item(), 2)
    
    return 0
