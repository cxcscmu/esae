import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import AutoModel, AutoTokenizer
from source.interface import Embedding


class MiniPcmEmbedding(Embedding):
    """
    This class implements the MiniPcm embedding model.
    """

    name = "MiniPcm"
    size = 2304

    def __init__(self, devices: List[int] = [0]) -> None:
        self.devices = devices
        assert len(self.devices) > 0
        self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-Embedding")
        self.pad_idx = self.tokenizer.pad_token_id
        kwargs = dict()
        kwargs["trust_remote_code"] = True
        kwargs["torch_dtype"] = torch.float16
        model = AutoModel.from_pretrained("openbmb/MiniCPM-Embedding", **kwargs)
        model = model.eval().to(devices[0])
        self.model = nn.DataParallel(model, devices)

    @torch.inference_mode()
    def forward(self, passages: List[str]) -> Tensor:
        """
        Adopted from https://huggingface.co/openbmb/MiniCPM-Embedding.
        """
        kwargs = dict()
        kwargs["padding"] = True
        kwargs["truncation"] = True
        kwargs["return_tensors"] = "pt"
        kwargs["return_attention_mask"] = True
        kwargs["max_length"] = 512
        encoded = self.tokenizer(passages, **kwargs)
        encoded = encoded.to(self.devices[0])
        outputs = self.model.forward(**encoded)
        masking = encoded["attention_mask"]
        s = torch.sum(outputs.last_hidden_state * masking.unsqueeze(-1).float(), dim=1)
        d = masking.sum(dim=1, keepdim=True).float()
        return F.normalize(s / d, p=2, dim=1).float()

    # @torch.inference_mode()
    # def forward_prefix(self, passages: List[str]) -> Tuple[Tensor, Any, Any]:
    #     """
    #     @todo: fix the return type.
    #     """
    #     kwargs = dict(padding=True, truncation=True, return_tensors="pt")
    #     encoded = self.tokenizer(passages[0], **kwargs)
    #     input_ids = encoded.input_ids[0]  # Shape: [seq_len]
    #     tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
    #     prefix_input_ids = [input_ids[:i] for i in range(1, len(input_ids) + 1)]
    #     batch_encoded = self.tokenizer.pad({'input_ids': prefix_input_ids}, padding=True, return_tensors="pt")
    #     batch_input_ids = batch_encoded.input_ids.to(self.devices[0])
    #     outputs = self.model(batch_input_ids)
    #     assert isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions)
    #     hiddens = outputs.last_hidden_state
    #     return F.normalize(hiddens[:, 0], p=2, dim=1), tokens, input_ids

    # @torch.inference_mode()
    # def forward_tokens(self, tokens: List[List[float]]) -> Tensor:
    #     kwargs = dict(padding=True, truncation=True, return_tensors="pt")
    #     batch_encoded = self.tokenizer.pad({'input_ids': tokens}, padding=True, return_tensors="pt")
    #     batch_input_ids = batch_encoded.input_ids.to(self.devices[0])
    #     outputs = self.model(batch_input_ids)
    #     assert isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions)
    #     hiddens = outputs.last_hidden_state
    #     return F.normalize(hiddens[:, 0], p=2, dim=1)
