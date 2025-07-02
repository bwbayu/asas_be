import torch.nn as nn
from transformers import AlbertConfig
from .modeling_albert_default import AlbertModel

class DirectSpecific(nn.Module):
    def __init__(self, model_name='indobenchmark/indobert-lite-base-p2'):
        super().__init__()
        # Load pretrained model
        self.config = AlbertConfig.from_pretrained(model_name)
        self.model = AlbertModel.from_pretrained(model_name, config=self.config)
        
        # Add regression layer
        self.dropout = nn.Dropout(p=0.3)
        self.regression_layer = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state

        cls_embedding = hidden_states[:, 0, :]    
        x = self.dropout(cls_embedding)
        score = self.regression_layer(x)
        return score