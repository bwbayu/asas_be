import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer

class SimilarityCross(nn.Module):
    def __init__(self, model_name='sentence-transformers/distiluse-base-multilingual-cased-v2'):
        super().__init__()
        # init pretrained
        self.model = SentenceTransformer(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)

        # add other component
        self.dense = nn.Linear(768, 512)
        # copy weight from pretrained model
        with torch.no_grad():
          self.dense.weight.copy_(self.model[2].linear.weight)
          self.dense.bias.copy_(self.model[2].linear.bias)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        embedding = hidden_states[:, 0, :] 
        x = self.dense(embedding)                                   # [B, 512]
        x = self.activation(x)
        return x