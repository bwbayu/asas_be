from .modeling_direct import DirectModel
from transformers import BertTokenizer
import torch
import warnings
import regex as re
import os
import boto3

warnings.simplefilter("ignore")
MODEL_NAME = 'indobenchmark/indobert-lite-base-p2'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = DirectModel(MODEL_NAME)

# download model from s3
S3_BUCKET = "model-asas-bucket"
MODEL_DIR = "model_direct"

def download_model_from_s3(model_key: str, local_filename: str):
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, local_filename)

    if not os.path.exists(local_path):
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, model_key, local_path)
        print(f"Downloaded {model_key} from S3.")

    return local_path

model_specific_path = download_model_from_s3("model_direct/model_1.pt", "model_1.pt")

# Load weight ke model
checkpoint_specific = torch.load(model_specific_path, map_location='cpu')
model.load_state_dict(checkpoint_specific)
model.eval()

def preprocess_text(text):
    text = text.lower()  # Ubah ke lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text)  # Hapus karakter khusus
    text = ' '.join(text.split())  # Hapus spasi berlebih
    return text

def get_score_direct(answer: str, reference: str, scenario: str):
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

# print(get_score_direct("Buah mengandung banyak vitamin seperti vitamin C yang bisa meningkatkan sistem imun, jadi tubuh tidak mudah sakit.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_direct("Buah itu penting karena bisa bikin kenyang dan segar, jadi kita nggak perlu makan makanan berat.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_direct("Makan buah bisa bikin tubuh jadi kurus karena buah mengandung banyak lemak yang dibakar saat tidur.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_direct("kami bangsa indonesia dengan ini menyatakan kemerdekaannya hal hal yang mengenai pemindahan kekuasaan dan lain lain di selenggarakan dengan cara seksama dan dengan tempo yang sesingkat singkatnya", "proklamasi kami bangsa indonesia dengan ini menyatakan kemerdekaan indonesia hal hal yang mengenai pemindahan kekuasaan d l l diselenggarakan dengan cara seksama dan dalam tempo yang sesingkat singkatnya jakarta 17 8 05 wakil wakil bangsa indonesia", "specific-prompt"))
