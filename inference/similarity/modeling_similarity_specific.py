import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer

class SimilaritySpecific(nn.Module):
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

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        embedding = self.mean_pooling(hidden_states, attention_mask)
        x = self.dense(embedding)                                   # [B, 512]
        x = self.activation(x)
        return x