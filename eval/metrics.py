import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_providers import BaseLLMProvider
import os
import json
from openai import OpenAI

# Download required NLTK data if not already present
for resource in ['tokenizers/punkt', 'corpora/wordnet', 'corpora/omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        download_name = resource.split('/')[-1]
        nltk.download(download_name)


class BleuMetric:
    def __init__(self):
        self.smoother = SmoothingFunction().method1

    def compute(self, reference: str, hypothesis: str) -> float:
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self.smoother)


class MeteorMetric:
    def compute(self, reference: str, hypothesis: str) -> float:
        # METEOR expects tokenized inputs. 
        # It expects a list of references (we provide one) and a single hypothesis.
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # If the hypothesis is completely empty/invalid, return 0 to avoid errors
        if not hyp_tokens:
            return 0.0
            
        return meteor_score([ref_tokens], hyp_tokens)


class RougeMetric:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute(self, reference: str, hypothesis: str) -> dict:
        scores = self.scorer.score(reference, hypothesis)
        return {
            "rouge1_f1": scores['rouge1'].fmeasure,
            "rouge2_f1": scores['rouge2'].fmeasure,
            "rougeL_f1": scores['rougeL'].fmeasure
        }


class BertScoreMetric:
    def __init__(self):
        # We load this once during initialization so it doesn't reload the model for every sentence.
        # rescale_with_baseline=True normalizes the scores to a more human-readable range (e.g., closer to 0-1)
        self.scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def compute(self, reference: str, hypothesis: str) -> dict:
        if not hypothesis.strip():
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        # BERTScorer expects lists of strings
        p, r, f1 = self.scorer.score([hypothesis], [reference])
        
        return {
            "precision": p.item(),
            "recall": r.item(),
            "f1": f1.item()
        }

class SimCSEMetric:
    def __init__(self, model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased"):
        # The model is downloaded from Hugging Face and cached locally
        self.model = SentenceTransformer(model_name)

    def compute(self, reference: str, hypothesis: str) -> float:
        # Handle empty predictions gracefully
        if not hypothesis.strip() or not reference.strip():
            return 0.0
            
        # 1. Convert both sentences into 768-dimensional dense vectors
        embeddings = self.model.encode([reference, hypothesis])
        
        # 2. Calculate the cosine similarity between the two vectors
        # embedding[0] is the reference, embedding[1] is the hypothesis
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])
        
        # Returns a float between -1.0 (opposite meaning) and 1.0 (identical meaning)
        return float(sim[0][0])

class LLMJudgeMetric:
    def __init__(self, provider: BaseLLMProvider = None):
        # Default to OpenAI if no provider is injected
        self.provider = provider 

    def compute(self, question: str, reference: str, hypothesis: str) -> dict:
        # Prompt exactly adapted from the LLM-as-a-Judge paper guidelines
        # for reference-guided single-answer evaluation.
        prompt = f"""
        [System]
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. 
        Your evaluation should consider correctness and helpfulness. 
        You will be given a reference answer and the assistant's answer.
        
        Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible.
        
        [User Question]
        {question}
        
        [The Start of Reference Answer]
        {reference}
        [The End of Reference Answer]
        
        [The Start of Assistant's Answer]
        {hypothesis}
        [The End of Assistant's Answer]
        
        After providing your explanation, you must rate the response on a scale of 1 to 10.
        Respond ONLY with a valid JSON object strictly following this format: 
        {{"score": <int>, "reasoning": "<string>"}}
        """
        
        return self.provider.generate_json(prompt)