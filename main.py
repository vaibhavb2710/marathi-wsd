# ============================================================
# main.py — Marathi WSD FastAPI Backend (complete fixed version)
# Run: uvicorn main:app --reload
# ============================================================

import json, re, torch, numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch import nn

# ============================================================
# CONFIG
# ============================================================
EXPORT_DIR = "marathi_wsd_deployment"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 128

# ============================================================
# MODEL DEFINITION
# Must be defined BEFORE loading weights
# ============================================================
class MuRILEncoder(nn.Module):
    def __init__(self, config_path):
        super().__init__()
        config = AutoConfig.from_pretrained(config_path)
        self.encoder = AutoModel.from_config(config)

        self.projector = nn.Sequential(
            nn.Linear(768, 512),   # projector.0
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)    # projector.3
        )
        self.classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        out  = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return self.classifier(self.projector(pooled))

# ============================================================
# LOAD EVERYTHING — order matters
# ============================================================
print("Loading metadata...")
with open(f"{EXPORT_DIR}/metadata.json", encoding="utf-8") as f:
    meta = json.load(f)

sense_gloss    = meta["sense_gloss"]
word_to_senses = meta["word_to_senses"]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(f"{EXPORT_DIR}/tokenizer")

print("Loading model...")
model = MuRILEncoder(f"{EXPORT_DIR}/encoder_config")
model.load_state_dict(
    torch.load(f"{EXPORT_DIR}/model_weights.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()
print(f"✅ Model loaded on {DEVICE}")


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Marathi WSD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at /ui
@app.get("/ui")
def serve_ui():
    return FileResponse("index.html")

# ============================================================
# REQUEST SCHEMA
# ============================================================
class PredictRequest(BaseModel):
    sentence: str

# ============================================================
# PREDICTION LOGIC
# ============================================================
def run_prediction(sentence: str):
    tokens      = re.findall(r'[\u0900-\u097F]+', sentence)
    found_words = list({
        word
        for token in tokens
        for word in word_to_senses
        if token.startswith(word)
    })

    if not found_words:
        return {
            "sentence": sentence,
            "results":  [],
            "error":    "No ambiguous word found in sentence"
        }

    results = []
    for target_word in found_words:
        candidates = word_to_senses.get(target_word, [])
        scores = {}

        for sid in candidates:
            gloss = sense_gloss.get(sid, "अर्थ")

            enc = tokenizer(
                sentence, gloss,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE)
                )
            prob = torch.softmax(logits, dim=1)[0][1].item()
            scores[sid] = round(prob, 4)

        best_sid      = max(scores, key=scores.get)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results.append({
            "word":             target_word,
            "predicted_sense":  best_sid,
            "predicted_gloss":  sense_gloss.get(best_sid, ""),
            "all_senses": [
                {
                    "sense_id": sid,
                    "score":    score,
                    "gloss":    sense_gloss.get(sid, ""),
                    "is_best":  sid == best_sid
                }
                for sid, score in sorted_scores
            ]
        })

    return {"sentence": sentence, "results": results, "error": None}

# ============================================================
# ROUTES
# ============================================================
@app.get("/")
def root():
    return {"status": "Marathi WSD API running"}

@app.get("/words")
def get_words():
    return {"words": list(word_to_senses.keys())}

@app.post("/predict")
def predict(req: PredictRequest):
    return run_prediction(req.sentence.strip())