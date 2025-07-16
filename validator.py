import json
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from typing import Optional

from guardrails.validators import Validator
from guardrails.validators import register_validator
from guardrails.validators import FailResult, PassResult

import litellm 

import logging

# Create named logger
logger = logging.getLogger("validator")
logger.setLevel(logging.INFO)  # Or DEBUG for more detail

# Avoid adding multiple handlers
if not logger.handlers:
    file_handler = logging.FileHandler("validator.log", mode='a')  # Relative to project root
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

logger.info("Logging initialized successfully.")

# Init NER
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
NER = pipeline("ner", model=model, tokenizer=tokenizer)

@register_validator(name="check_competitor_mentions", data_type="string")
class CheckCompetitorMentions(Validator):
    def __init__(self, competitors: List[str], **kwargs):
        self.competitors = competitors
        self.competitors_lower = [comp.lower() for comp in competitors]
        self.ner = NER
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.competitor_embeddings = self.sentence_model.encode(self.competitors)
        self.similarity_threshold = 0.6
        super().__init__(**kwargs)

    def exact_match(self, text: str) -> List[str]:
        text_lower = text.lower()
        matches = []
        for comp, comp_lower in zip(self.competitors, self.competitors_lower):
            if comp_lower in text_lower:
                if re.search(r'\b' + re.escape(comp_lower) + r'\b', text_lower):
                    matches.append(comp)
        return matches

    def extract_entities(self, text: str) -> List[str]:
        ner_results = self.ner(text)
        entities = []
        current_entity = ""
        for item in ner_results:
            if item['entity'].startswith('B-'):
                if current_entity:
                    entities.append(current_entity.strip())
                current_entity = item['word']
            elif item['entity'].startswith('I-'):
                current_entity += " " + item['word']
        if current_entity:
            entities.append(current_entity.strip())
        return entities

    def vector_similarity_match(self, entities: List[str]) -> List[str]:
        if not entities:
            return []
        entity_embeddings = self.sentence_model.encode(entities)
        similarities = cosine_similarity(entity_embeddings, self.competitor_embeddings)
        matches = []
        for i, entity in enumerate(entities):
            max_similarity = np.max(similarities[i])
            if max_similarity >= self.similarity_threshold:
                most_similar_competitor = self.competitors[np.argmax(similarities[i])]
                matches.append(most_similar_competitor)
        return matches


    def llm_detect_competitor(self, text: str, model_name: str = None) -> Optional[str]:
        if not text.strip():
            logger.warning("Empty input received for LLM detection. Skipping.")
            return None

        system_prompt = """
    You are a strict JSON-only filter that checks if the input mentions any competitors of Bookme.pk.

    You MUST respond in the following exact format:
    {
    "competitors": ["competitor1", "competitor2"]
    }

    Rules:
    - Always use lowercase for competitor names.
    - If no competitor is mentioned, respond with: { "competitors": [] }
    - NEVER include any explanation, greeting, or text outside the JSON block.
    - You are Filter of Bookme only focus on competitors of bookme

    Examples:
    Input: "I used EasyTickets" → {"competitors": ["easytickets"]}
    Input: "Bookme.pk is great" → {"competitors": []}
    Input: "tripadvisor and bookme are good" → {"competitors": ["tripadvisor"]}
    Input: "" → {"competitors": []}
    """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        logger.info(f"LLM detection input: {text}")

        try:
            result = litellm.completion(
                model=model_name or "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=messages,
                max_tokens=100,
                temperature=0.0
            )

            response = result.choices[0].message.content.strip()
            logger.info(f"LLM raw response: {response}")

            # Attempt to parse the JSON
            try:
                parsed = json.loads(response)
                competitors = parsed.get("competitors", [])
                if not isinstance(competitors, list):
                    raise ValueError("Invalid format for 'competitors' field")
            except Exception as parse_err:
                logger.error(f"Failed to parse LLM response as JSON: {parse_err}")
                return None

            # Filter out bookme variants and empty values
            filtered = [
                c for c in competitors
                if c and "bookme" not in c and c not in {"none", "n/a"}
            ]

            if not filtered:
                return None

            logger.info(f"LLM detected competitors: {filtered}")
            return ", ".join(filtered)

        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return None


    def validate(self, value: str, metadata: Optional[dict[str, str]] = None):
        model_used = metadata.get("model") if metadata else None
        print(f"Running validator. Model: {model_used or 'N/A'}")

        # Step 1: Exact matching
        exact_matches = self.exact_match(value.lower())
        if exact_matches:
            logger.info(f"Found exact matches: {exact_matches}")
            return FailResult(
                error_message=f"Mentions competitors: {', '.join(exact_matches)}"
            )

        # Step 2: Entity extraction + vector similarity
        entities = self.extract_entities(value)
        similarity_matches = self.vector_similarity_match(entities)

        if similarity_matches:
            logger.info(f"Found similarity matches: {similarity_matches}")
            return FailResult(
                error_message=f"Mentions competitors (via NER): {', '.join(similarity_matches)}"
            )

        #Step 3: LLM detection
        llm_result = self.llm_detect_competitor(text=value, model_name=model_used)
        if llm_result:
            logger.info(f"LLM detected competitors: {llm_result}")
            return FailResult(
                error_message=f"LLM detected competitor mention: {llm_result}"
            )
        return PassResult()