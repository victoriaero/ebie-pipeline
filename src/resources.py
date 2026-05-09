from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, RobertaForMaskedLM, RobertaTokenizer


DEFAULT_CLASSIFIER_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_CLASSIFIER_TARGET_CLASS = "joy"


@dataclass
class ModelResources:
    device: torch.device
    tokenizer: object
    model: object
    model_max_length: int
    classifier_tokenizer: object
    classifier_model: object


def tokenize_for_roberta(resources, text):
    return resources.tokenizer(text, return_tensors="pt", truncation=True, max_length=resources.model_max_length,).to(resources.device)


def _resolve_label_id(classifier_model, target_class):
    id2label = getattr(classifier_model.config, "id2label", {}) or {}
    normalized_target = str(target_class).strip().lower()
    for label_id, label_name in id2label.items():
        if str(label_name).strip().lower() == normalized_target:
            return int(label_id)

    available_labels = ", ".join(str(label) for label in id2label.values())
    raise ValueError(f"Target classifier class '{target_class}' was not found in classifier labels: " f"{available_labels}")


def load_resources(config):
    if config["use_gpu_if_available"] and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = RobertaTokenizer.from_pretrained(config["roberta_model_name"])
    model = RobertaForMaskedLM.from_pretrained(config["roberta_model_name"]).to(device)
    model.eval()
    model_max_length = min(tokenizer.model_max_length, model.config.max_position_embeddings)

    classifier_model_name = config.get("classifier_model_name") or DEFAULT_CLASSIFIER_MODEL_NAME
    config["classifier_model_name"] = classifier_model_name
    config["emotion_model_name"] = classifier_model_name
    config.setdefault("classifier_target_class", DEFAULT_CLASSIFIER_TARGET_CLASS)

    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_name).to(device)
    classifier_model.eval()
    config["classifier_target_label"] = _resolve_label_id(classifier_model, config["classifier_target_class"],)

    return ModelResources(device=device, tokenizer=tokenizer, model=model, model_max_length=model_max_length, classifier_tokenizer=classifier_tokenizer, classifier_model=classifier_model,)
