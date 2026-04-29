from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForMaskedLM, RobertaTokenizer


@dataclass
class ModelResources:
    device: torch.device
    tokenizer: object
    model: object
    model_max_length: int
    classifier_tokenizer: object
    classifier_model: object


def tokenize_for_roberta(resources, text):
    return resources.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=resources.model_max_length,
    ).to(resources.device)


def load_resources(config):
    if config["use_gpu_if_available"] and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = RobertaTokenizer.from_pretrained(config["roberta_model_name"])
    model = RobertaForMaskedLM.from_pretrained(config["roberta_model_name"]).to(device)
    model.eval()
    model_max_length = min(tokenizer.model_max_length, model.config.max_position_embeddings)

    classifier_model_name = config.get("classifier_model_name", config.get("emotion_model_name"))

    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        classifier_model_name
    ).to(device)
    classifier_model.eval()

    return ModelResources(
        device=device,
        tokenizer=tokenizer,
        model=model,
        model_max_length=model_max_length,
        classifier_tokenizer=classifier_tokenizer,
        classifier_model=classifier_model,
    )
