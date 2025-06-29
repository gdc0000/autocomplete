from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import spacy
import json
from pathlib import Path

# Carica spaCy
nlp = spacy.load("en_core_web_sm")

# Parole grammaticali sempre presenti
ALWAYS_PRESENT_GRAMMAR = {
    "articles": ["a", "an", "the, what"],
    "pronouns": ["I", "you", "we", "me", "it", "they", "your", "my", "our", "us", "them"],
    "prepositions": ["in", "on", "at", "to", "with", "from", "about", "for", "of", "by", "as", "how"],
    "particles": ["to", "not"],
    "conjunctions": ["and", "but", "or", "so", "because", "although", "if"],
    "auxiliaries": ["is", "are", "was", "were", "be", "being", "been", "do", "did", "does", "will", "can", "could", "should", "would", "may", "might", "must", "shall"]
}

# Parole chiave di contenuto da file JSON
KEYWORDS_PATH = Path(__file__).parent / "prompt_keywords.json"
with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
    KEYWORDS = json.load(f)

# Inizializzazione FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caricamento modello GPT2
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

# Input schema
class PromptInput(BaseModel):
    text: str
    max_words: int = 5

# Funzione di pulizia e ordinamento
def clean_and_sort(lst):
    return sorted(set(w.lower() for w in lst if w.isalpha()))

@app.post("/suggest")
def suggest_words(prompt: PromptInput):
    input_ids = tokenizer.encode(prompt.text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=50)

    raw_words = [tokenizer.decode([i.item()]).strip() for i in top_indices]
    doc = nlp(" ".join(raw_words))

    nouns, adjectives, verbs, others = [], [], [], []

    for token in doc:
        word = token.text
        if not word.isalpha() or len(word) < 2:
            continue
        if token.pos_ in ["NOUN", "PROPN"]:
            nouns.append(word)
        elif token.pos_ in ["ADJ", "ADV"]:
            adjectives.append(word)
        elif token.pos_ == "VERB" and token.lemma_ not in ALWAYS_PRESENT_GRAMMAR["auxiliaries"]:
            verbs.append(word)
        else:
            others.append(word)

    return {
        "nouns": clean_and_sort(nouns + KEYWORDS.get("objects", []))[:prompt.max_words],
        "adjectives": clean_and_sort(adjectives + KEYWORDS.get("qualities", []))[:prompt.max_words],
        "verbs": clean_and_sort(verbs + KEYWORDS.get("actions", []))[:prompt.max_words],
        "others": clean_and_sort(others + KEYWORDS.get("others", []))[:prompt.max_words],
        "articles": clean_and_sort(ALWAYS_PRESENT_GRAMMAR["articles"]),
        "pronouns": clean_and_sort(ALWAYS_PRESENT_GRAMMAR["pronouns"]),
        "prepositions": clean_and_sort(ALWAYS_PRESENT_GRAMMAR["prepositions"]),
        "particles": clean_and_sort(ALWAYS_PRESENT_GRAMMAR["particles"]),
        "conjunctions": clean_and_sort(ALWAYS_PRESENT_GRAMMAR["conjunctions"]),
        "auxiliaries": clean_and_sort(ALWAYS_PRESENT_GRAMMAR["auxiliaries"])
    }
