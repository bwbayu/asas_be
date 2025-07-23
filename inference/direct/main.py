from .modeling_direct import DirectModel
from transformers import BertTokenizer, AutoTokenizer
import torch
import warnings
import regex as re
import os

warnings.simplefilter("ignore")
MODEL_NAME = 'google-bert/bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_specific = DirectModel(MODEL_NAME).to('cuda')
model_cross = DirectModel(MODEL_NAME).to('cuda')

# Load weight ke model
BASE_DIR = os.path.dirname(__file__)
checkpoint_specific = torch.load(os.path.join(BASE_DIR, 'model', 'model_2s.pt'), map_location='cuda')
model_specific.load_state_dict(checkpoint_specific)
model_specific.eval()

checkpoint_cross = torch.load(os.path.join(BASE_DIR, 'model', 'model_2c.pt'), map_location='cuda')
model_cross.load_state_dict(checkpoint_cross)
model_cross.eval()

models = {
    "specific-prompt": model_specific,
    "cross-prompt": model_cross
}

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

    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        model = models[scenario].to('cuda')
        predictions = model(**inputs).squeeze(1)
        score = torch.clamp(predictions, 0, 1)
        return round(score.item(), 2)
    
    return 0
