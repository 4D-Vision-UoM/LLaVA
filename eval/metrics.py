import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, models
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
        # 1. Load the raw transformer model
        word_embedding_model = models.Transformer(model_name)
        
        # 2. Force the pooling layer to use the [CLS] token (as intended by SimCSE)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode='cls'  # <-- The crucial fix
        )
        
        # 3. Combine them into a proper SentenceTransformer
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def compute(self, reference: str, hypothesis: str) -> float:
        if not hypothesis.strip() or not reference.strip():
            return 0.0
            
        embeddings = self.model.encode([reference, hypothesis])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])
        
        return float(sim[0][0])

class LLMJudgeMetric:
    def __init__(self, provider: BaseLLMProvider = None):
        # Default to OpenAI if no provider is injected
        self.provider = provider 

    def compute(self, question: str, reference: str, hypothesis: str) -> dict:
        # Prompt exactly adapted from the LLM-as-a-Judge paper guidelines
        # for reference-guided single-answer evaluation.
        safe_reference = json.dumps(reference)
        safe_hypothesis = json.dumps(hypothesis)
        print(f"\nEvaluating with LLM Judge...\nQuestion: {question}\nReference: {reference}\nHypothesis: {hypothesis}")    
        
        prompt = f"""
        [System]
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. 
        Your evaluation should consider correctness and helpfulness. 
        You will be given a reference answer and the assistant's answer.
        
        Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in
        which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
        
        [User Question]
        {question}
        
        [The Start of Reference Answer]
        {safe_reference}
        [The End of Reference Answer]
        
        [The Start of Assistant's Answer]
        {safe_hypothesis}
        [The End of Assistant's Answer]
        
        After providing your explanation, you must rate the response on a scale of 1 to 10.
        Respond ONLY with a valid JSON object strictly following this format: 
        {{"score": <int>, "reasoning": "<string>"}}
        """
        
        return self.provider.generate_json(prompt)