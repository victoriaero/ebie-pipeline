from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForMaskedLM, RobertaTokenizer


@dataclass
class ModelResources:
    device: torch.device
    tokenizer: object
    model: object
    emotion_tokenizer: object
    emotion_model: object


def load_resources(config):
    if config["use_gpu_if_available"] and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = RobertaTokenizer.from_pretrained(config["roberta_model_name"])
    model = RobertaForMaskedLM.from_pretrained(config["roberta_model_name"]).to(device)
    model.eval()

    emotion_tokenizer = AutoTokenizer.from_pretrained(config["emotion_model_name"])
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        config["emotion_model_name"]
    ).to(device)
    emotion_model.eval()

    return ModelResources(
        device=device,
        tokenizer=tokenizer,
        model=model,
        emotion_tokenizer=emotion_tokenizer,
        emotion_model=emotion_model,
    )
