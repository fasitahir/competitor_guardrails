from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

from guardrails.validators import Validator
from guardrails.validators import register_validator
from guardrails.validators import FailResult, PassResult

import litellm  # <--- using this to call your LLM
import os

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

    def llm_detect_competitor(text: str, model_name: str = None) -> Optional[str]:
        system_prompt = (
            "Check whether the input text mentions any ticketing or booking companies other than Bookme.pk. "
            "Return exactly 'None' if no competitors are detected. "
            "If competitors are detected, return their names in lowercase, separated by commas, with no additional text. "
            "Exclude Bookme.pk and its variations (e.g., 'bookme', 'book me'). "
            "DO NOT include explanations, periods, or any other text. "
            "Examples: 'None', 'easytickets,tripadvisor'"
            "Example use cases: " \
            "If user mentions 'bookme', return 'None'. " \
            "If user mentions 'easytickets', return 'easytickets'. "
            "If user mentions 'bookme and tripadvisor', return 'tripadvisor'. " \
            "If user mentions 'bookme.pk and easytickets', return 'easytickets. "
            'User enters '' returns None. ' \
            "Do not return any other sentence or even word Just return the competitor names or 'None'. "

        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        try:
            result = litellm.completion(
                model=model_name or "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=messages,
                max_tokens=100,
                temperature=0.0
            )

            response = result.choices[0].message.content.strip().lower()

            # Check if response starts with "none" or is empty
            if response.startswith("none") or not response:
                return None

            # Explicitly filter out Bookme-related terms
            if any(term in response for term in {"bookme", "bookme.pk", "book me"}):
                return None

            # Split response into potential competitor names
            competitors = [name.strip() for name in response.split(",") if name.strip()]
            filtered_competitors = [
                name for name in competitors 
                if name and "bookme" not in name and name not in {"none", "no", "n/a", ""}
            ]

            # Return None if no valid competitors remain
            if not filtered_competitors:
                return None

            return ", ".join(filtered_competitors)

        except Exception as e:
            print(f"⚠️ LLM detection failed: {e}")
            return None


    def validate(self, value: str, metadata: Optional[dict[str, str]] = None):
        model_used = metadata.get("model") if metadata else None
        print(f"🔍 Running validator. Model: {model_used or 'N/A'}")

        # Step 1: Exact matching
        exact_matches = self.exact_match(value)
        if exact_matches:
            return FailResult(
                error_message=f"Mentions competitors: {', '.join(exact_matches)}"
            )

        # Step 2: Entity extraction + vector similarity
        entities = self.extract_entities(value)
        similarity_matches = self.vector_similarity_match(entities)

        if similarity_matches:
            return FailResult(
                error_message=f"Mentions competitors (via NER): {', '.join(similarity_matches)}"
            )

        # #Step 3: LLM detection
        # llm_result = self.llm_detect_competitor(value, model_name=model_used)
        # if llm_result:
        #     return FailResult(
        #         error_message=f"LLM detected competitor mention: {llm_result}"
        #     )

        return PassResult()