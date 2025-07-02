from .modeling_direct_specific import DirectSpecific
from .modeling_direct_cross import DirectCross
from transformers import BertTokenizer
import torch
import warnings
import regex as re
import os

warnings.simplefilter("ignore")
MODEL_NAME = 'indobenchmark/indobert-lite-base-p2'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model_specific = DirectSpecific(MODEL_NAME)
model_cross = DirectCross(MODEL_NAME)

# Load weight ke model
BASE_DIR = os.path.dirname(__file__)
checkpoint_specific = torch.load(os.path.join(BASE_DIR, 'model', 'model_1.pt'), map_location='cpu')
model_specific.load_state_dict(checkpoint_specific)
model_specific.eval()

checkpoint_cross = torch.load(os.path.join(BASE_DIR, 'model', 'model_10.pt'), map_location='cpu')
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

    with torch.no_grad():
        model = models[scenario]
        predictions = model(**inputs).squeeze(1)
        score = torch.clamp(predictions, 0, 1)
        return round(score.item(), 2)
    
    return 0

# print(get_score_direct("Buah mengandung banyak vitamin seperti vitamin C yang bisa meningkatkan sistem imun, jadi tubuh tidak mudah sakit.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_direct("Buah itu penting karena bisa bikin kenyang dan segar, jadi kita nggak perlu makan makanan berat.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_direct("Makan buah bisa bikin tubuh jadi kurus karena buah mengandung banyak lemak yang dibakar saat tidur.", "Mengonsumsi buah membantu menjaga daya tahan tubuh karena kaya akan vitamin, mineral, dan antioksidan yang dibutuhkan tubuh.", "specific-prompt"))
# print(get_score_direct("kami bangsa indonesia dengan ini menyatakan kemerdekaannya hal hal yang mengenai pemindahan kekuasaan dan lain lain di selenggarakan dengan cara seksama dan dengan tempo yang sesingkat singkatnya", "proklamasi kami bangsa indonesia dengan ini menyatakan kemerdekaan indonesia hal hal yang mengenai pemindahan kekuasaan d l l diselenggarakan dengan cara seksama dan dalam tempo yang sesingkat singkatnya jakarta 17 8 05 wakil wakil bangsa indonesia", "specific-prompt"))
