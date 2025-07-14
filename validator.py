from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

from guardrails.validators import Validator
from guardrails.validators import register_validator
from guardrails.validators import FailResult, PassResult

# Initialize NER pipeline
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
NER = pipeline("ner", model=model, tokenizer=tokenizer)

@register_validator(name="check_competitor_mentions", data_type="string")
class CheckCompetitorMentions(Validator):
    def __init__(
        self,
        competitors: List[str],
        **kwargs
    ):
        self.competitors = competitors
        self.competitors_lower = [comp.lower() for comp in competitors]

        self.ner = NER

        # Initialize sentence transformer for vector embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pre-compute competitor embeddings
        self.competitor_embeddings = self.sentence_model.encode(self.competitors)

        # Set the similarity threshold
        self.similarity_threshold = 0.6
        
        super().__init__(**kwargs)

    def exact_match(self, text: str) -> List[str]:
        text_lower = text.lower()
        matches = []
        for comp, comp_lower in zip(self.competitors, self.competitors_lower):
            if comp_lower in text_lower:
                # Use regex to find whole word matches
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
    
    def validate(
        self,
        value: str,
        metadata: Optional[dict[str, str]] = None
    ):
        # Step 1: Perform exact matching on the entire text
        exact_matches = self.exact_match(value)
        
        if exact_matches:
            return FailResult(
                error_message=f"Your response directly mentions competitors: {', '.join(exact_matches)}"
            )

        # Step 2: Extract named entities
        entities = self.extract_entities(value)

        # Step 3: Perform vector similarity matching
        similarity_matches = self.vector_similarity_match(entities)

        # Step 4: Combine matches and check if any were found
        all_matches = list(set(exact_matches + similarity_matches))

        if all_matches:
            return FailResult(
                error_message=f"Your response mentions competitors: {', '.join(all_matches)}"
            )

        return PassResult()      
    
