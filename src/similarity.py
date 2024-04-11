"""
Модуль расчета близости изображений, текстов
"""

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    pipeline,
)
import torch
import numpy as np


def image_similarity(
    image_1: np.ndarray,
    image_2: np.ndarray,
    model_ckpt: str = "google/vit-base-patch16-224-in21k",
) -> float:
    """Computes cosine similarity between two images.
    Original article: https://huggingface.co/blog/image-similarity

    Args:
        image_1 (np.ndarray): image, like cv2.imread()
        image_2 (np.ndarray): image, like cv2.imread()
        model_ckpt (str, optional): name of transformers model.
        Defaults to "google/vit-base-patch16-224-in21k".

    Returns:
        float: value of similarity
    """

    def compute_scores(emb_one, emb_two):
        """Computes cosine similarity between two vectors."""
        scores = torch.nn.functional.cosine_similarity(
            torch.sum(emb_one, dim=1), torch.sum(emb_two, dim=1)
        )
        return scores.numpy().tolist()[0]

    extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    inputs = extractor(image_1, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    emb_one = outputs.last_hidden_state

    inputs = extractor(image_2, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    emb_two = outputs.last_hidden_state

    score = compute_scores(emb_one, emb_two)
    return score


def text_similarity(
    query: str,
    description: str,
    model_name_sentience: str = "sentence-transformers/all-mpnet-base-v2",
) -> dict[str, list[tuple[float, str]]]:
    """Calculate similarity between texts: "query" and "description".
    If they doesn't have any NER-entities return 0.
    Else return similarity between each sentience from "query" (key of dict)
    and each sentience from "description" (value of dict:
    list of tuple(similarity <float>, sentience of "description" <str>))

    Args:
        query (str): short query to find similarity from description
        description (str): full text for finding query
        model_name_sentience (str, optional): name of feature extract model.
        Defaults to "sentence-transformers/all-mpnet-base-v2".
    """

    def get_entities(
        text: str, model_name_ner: str = "Babelscape/wikineural-multilingual-ner"
    ) -> set:
        """Return set with entities after applying NER-analisys

        Args:
            text (str): text which need to find entity
            model_name_ner (str, optional): model name of NER analisys.
              Defaults to "Babelscape/wikineural-multilingual-ner".

        Returns:
            set: entities in set()
        """
        tokenizer_ner = AutoTokenizer.from_pretrained(model_name_ner)
        model_ner = AutoModelForTokenClassification.from_pretrained(model_name_ner)

        ner = pipeline(
            "ner", model=model_ner, tokenizer=tokenizer_ner, grouped_entities=True
        )

        ner_results = ner(text)

        entities = set()
        for i, match in enumerate(ner_results):
            if ner_results[i]["start"] == ner_results[i - 1]["end"]:
                entities.add(
                    (ner_results[i - 1]["word"] + match["word"].split("##")[1])
                )
                continue
            if match["word"][:2] == "##":
                continue
            entities.add(match["word"])

        return entities

    def split_sentinces(text: str) -> list[str]:
        """Split sentincies on sentince, using strip and split function

        Args:
            text (str): raw text

        Returns:
            list[str]: list with sentincies after cleaning
        """
        result = []
        for line in text.strip().replace("\n", "").split("."):
            if line:
                result.append(line.strip())
        return result

    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    entities_query = get_entities(query)
    entities_description = get_entities(description)

    if not entities_query.intersection(entities_description):
        return 0

    # Encode text
    def encode(
        texts: list[str] | str, tokenizer: AutoTokenizer, model: AutoModel
    ) -> torch.Tensor:

        # Tokenize sentences
        encoded_input = tokenizer(
            texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input, return_dict=True)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        return sentence_embeddings

    results = {}

    docs = split_sentinces(description)
    queries = split_sentinces(query)

    tokenizer = AutoTokenizer.from_pretrained(model_name_sentience)
    model = AutoModel.from_pretrained(model_name_sentience)

    # Encode docs
    doc_emb = encode(docs, tokenizer, model)

    for query in queries:
        # Encode every query
        query_emb = encode(query, tokenizer, model)

        # Compute dot score between query and all document embeddings
        scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

        results[query] = []
        for doc, score in zip(scores, docs):
            results[query].append((doc, score))

        # Sort by decreasing score for each query
        results[query] = sorted(results[query], key=lambda x: x[0], reverse=True)

    return results
