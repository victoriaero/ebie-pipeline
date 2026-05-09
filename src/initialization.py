import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_LLM_CACHE = {}


def _generate_random_token(resources, config):
    del config

    while True:
        token_id = random.randrange(resources.tokenizer.vocab_size)
        if token_id in resources.tokenizer.all_special_ids:
            continue
        token = resources.tokenizer.decode([token_id], skip_special_tokens=True).strip()
        if token:
            return token


def _generate_random_phrase(resources, config, num_tokens):
    return " ".join(_generate_random_token(resources, config) for _ in range(num_tokens))


def _load_llm(resources, config):
    cache_key = (config["llm_initialization_model"], resources.device.type)
    if cache_key not in _LLM_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(config["llm_initialization_model"])
        model = AutoModelForCausalLM.from_pretrained(config["llm_initialization_model"]).to(resources.device)
        model.eval()
        _LLM_CACHE[cache_key] = (tokenizer, model)
    return _LLM_CACHE[cache_key]


def _extract_candidate_words(text):
    candidates = []
    for item in re.findall(r"[A-Za-z][A-Za-z\-]*", text):
        normalized = item.strip().lower()
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    return candidates


def _generate_llm_word_bank(resources, config):
    tokenizer, model = _load_llm(resources, config)
    prompt = (
        "Provide a short list of English words or a very short phrase strongly related to "
        f"the class '{config['llm_initialization_target_class']}'. "
        "Return only comma-separated words or short phrases, with no explanation."
    )
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        model_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,)
    else:
        model_input = prompt

    inputs = tokenizer(model_input, return_tensors="pt").to(resources.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=config["llm_initialization_max_new_tokens"], do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id,)

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,)
    candidates = _extract_candidate_words(generated_text)
    if not candidates:
        raise ValueError("The LLM initialization did not return usable words.")
    return candidates


def _generate_llm_phrase(resources, config):
    word_bank = _generate_llm_word_bank(resources, config)
    num_words = min(max(1, config["llm_initialization_num_words"]), len(word_bank))
    return " ".join(random.sample(word_bank, num_words))


def generate_initial_population(resources, config, size):
    if config["initialization_mode"] == "random_1_token":
        return [_generate_random_phrase(resources, config, 1) for _ in range(size)]

    if config["initialization_mode"] == "random_3_tokens":
        return [_generate_random_phrase(resources, config, 3) for _ in range(size)]

    if config["initialization_mode"] == "llm_tokens":
        return [_generate_llm_phrase(resources, config) for _ in range(size)]

    raise ValueError(f"Unsupported initialization mode: {config['initialization_mode']}")
