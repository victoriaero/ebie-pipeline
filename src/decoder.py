import torch
import torch.nn.functional as F


def decode_com_top_k(logits, top_k):
    tokens_selecionados = []
    for logit in logits[0]:
        top_k_values, top_k_indices = torch.topk(logit, top_k)
        probs = F.softmax(top_k_values, dim=-1)
        indice_selecionado = torch.multinomial(probs, num_samples=1).item()
        tokens_selecionados.append(top_k_indices[indice_selecionado].item())
    return tokens_selecionados


def _mask_special_tokens(tokenizer, scores, filter_special_tokens):
    if not filter_special_tokens:
        return scores

    masked_scores = scores.clone()
    if tokenizer.all_special_ids:
        masked_scores[tokenizer.all_special_ids] = -torch.inf
    return masked_scores


def _get_scores_from_lm_head(resources, token_embedding):
    return resources.model.lm_head(token_embedding.unsqueeze(0).unsqueeze(0))[0, 0]


def _get_scores_from_embedding_similarity(resources, token_embedding, similarity):
    embedding_matrix = resources.model.get_input_embeddings().weight

    if similarity == "cosine":
        normalized_embedding = F.normalize(token_embedding.unsqueeze(0), dim=-1)
        normalized_matrix = F.normalize(embedding_matrix, dim=-1)
        return torch.matmul(normalized_matrix, normalized_embedding.squeeze(0))

    return torch.matmul(embedding_matrix, token_embedding)


def _select_token_id(resources, scores, strategy, top_k, filter_special_tokens):
    filtered_scores = _mask_special_tokens(resources.tokenizer, scores, filter_special_tokens)
    top_scores, top_indices = torch.topk(filtered_scores, top_k)

    if strategy == "argmax":
        return top_indices[0].item()

    if strategy == "weighted_sampling":
        probabilities = F.softmax(top_scores, dim=-1)
        selected_index = torch.multinomial(probabilities, num_samples=1).item()
        return top_indices[selected_index].item()

    raise ValueError(f"Unsupported decoder strategy: {strategy}")


def decode_embeddings_to_token_ids(resources, config, embeddings):
    token_ids = []
    for token_embedding in embeddings[0]:
        if config["decoder_family"] == "lm_head":
            scores = _get_scores_from_lm_head(resources, token_embedding)
        elif config["decoder_family"] == "embedding_similarity":
            scores = _get_scores_from_embedding_similarity(
                resources,
                token_embedding,
                config["decoder_similarity"],
            )
        else:
            raise ValueError(f"Unsupported decoder family: {config['decoder_family']}")

        token_ids.append(
            _select_token_id(
                resources,
                scores,
                config["decoder_strategy"],
                config["decoder_top_k"],
                config["filter_special_tokens"],
            )
        )
    return token_ids


def decode_embeddings_to_text(resources, config, embeddings):
    token_ids = decode_embeddings_to_token_ids(resources, config, embeddings)
    return resources.tokenizer.decode(token_ids, skip_special_tokens=True)
