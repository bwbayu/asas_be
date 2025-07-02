import torch
import torch.nn as nn
from transformers import AlbertConfig
from .modeling_albert_default import AlbertModel

class DirectCross(nn.Module):
    def __init__(self, model_name='indobenchmark/indobert-lite-base-p2'):
        super().__init__()
        # Load pretrained model
        self.config = AlbertConfig.from_pretrained(model_name)
        self.model = AlbertModel.from_pretrained(model_name, config=self.config)
        
        # Add regression layer
        self.dropout = nn.Dropout(p=0.3)
        self.regression_layer = nn.Linear(self.config.hidden_size, 1)
        self.query_vector = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state

        attn_scores = self.query_vector(hidden_states).squeeze(-1)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        embedding = torch.sum(hidden_states * attn_weights, dim=1)  
        x = self.dropout(embedding)
        score = self.regression_layer(x)
        return score