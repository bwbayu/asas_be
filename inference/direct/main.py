from .modeling_direct import DirectModel
from transformers import BertTokenizer
import torch
import warnings
import regex as re
import os

warnings.simplefilter("ignore")
MODEL_NAME = 'indobenchmark/indobert-lite-base-p2'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = DirectModel(MODEL_NAME)

# Load weight ke model
BASE_DIR = os.path.dirname(__file__)
checkpoint_specific = torch.load(os.path.join(BASE_DIR, 'model', 'model_1.pt'),  map_location=torch.device('cpu'))
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
