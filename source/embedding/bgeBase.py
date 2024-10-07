import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from source.interface import Embedding


class BgeBaseEmbedding(Embedding):
    """
    This class implements the BgeBase embedding model.
    """

    name = "BgeBase"
    size = 768

    def __init__(self, devices: List[int] = [0]) -> None:
        self.devices = devices
        assert len(self.devices) > 0
        self.tokenizer = BertTokenizerFast.from_pretrained("BAAI/bge-base-en-v1.5")
        self.pad_idx = self.tokenizer.pad_token_id
        assert isinstance(self.tokenizer, BertTokenizerFast)
        model = BertModel.from_pretrained("BAAI/bge-base-en-v1.5")
        assert isinstance(model, BertModel)
        model = model.eval().to(devices[0])
        self.model = nn.DataParallel(model, devices)

    @torch.inference_mode()
    def forward(self, passages: List[str]) -> Tensor:
        kwargs = dict(padding=True, truncation=True, return_tensors="pt")
        encoded = self.tokenizer(passages, **kwargs)
        outputs = self.model.forward(**encoded.to(self.devices[0]))
        assert isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions)
        hiddens = outputs.last_hidden_state
        return F.normalize(hiddens[:, 0], p=2, dim=1)
    
    @torch.inference_mode()
    def forward_prefix(self, passages: List[str]) -> Tuple[Tensor, Any, Any]:
        """
        @todo: fix the return type.
        """
        kwargs = dict(padding=True, truncation=True, return_tensors="pt")
        encoded = self.tokenizer(passages[0], **kwargs)
        input_ids = encoded.input_ids[0]  # Shape: [seq_len]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        prefix_input_ids = [input_ids[:i] for i in range(1, len(input_ids) + 1)]
        batch_encoded = self.tokenizer.pad({'input_ids': prefix_input_ids}, padding=True, return_tensors="pt")
        batch_input_ids = batch_encoded.input_ids.to(self.devices[0])
        outputs = self.model(batch_input_ids)
        assert isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions)
        hiddens = outputs.last_hidden_state
        return F.normalize(hiddens[:, 0], p=2, dim=1), tokens, input_ids

    @torch.inference_mode()
    def forward_tokens(self, tokens: List[List[float]]) -> Tensor:
        kwargs = dict(padding=True, truncation=True, return_tensors="pt")
        batch_encoded = self.tokenizer.pad({'input_ids': tokens}, padding=True, return_tensors="pt")
        batch_input_ids = batch_encoded.input_ids.to(self.devices[0])
        outputs = self.model(batch_input_ids)
        assert isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions)
        hiddens = outputs.last_hidden_state
        return F.normalize(hiddens[:, 0], p=2, dim=1)
