from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from flask import render_template
from sentence_transformers import SentenceTransformer, util


# Words we DON'T want as keywords
STOP_WORDS = {
    "using", "based", "approach", "method", "analysis", "study",
    "system", "model", "technique", "paper", "research", "work"
}

def clean_keywords(keywords):
    cleaned = []

    for word, score in keywords:
        word_lower = word.lower().strip()

        # remove very short words
        if len(word_lower) < 4:
            continue

        # remove keywords containing stop words inside phrase
        if any(stop in word_lower.split() for stop in STOP_WORDS):
            continue

        cleaned.append(word_lower)

    # Remove single words if part of a bigger phrase
    final_keywords = []
    for word in cleaned:
        if not any(word != other and word in other for other in cleaned):
            final_keywords.append(word)

    return list(set(final_keywords))

def expand_keywords(base_keywords):
    # A small research concept bank (we will grow this later)
    concept_bank = [
        "medical image segmentation",
        "glioma classification",
        "radiology AI",
        "convolutional neural networks in healthcare",
        "MRI scan analysis",
        "tumor localization",
        "computer aided diagnosis",
        "biomedical image processing",
        "neural network based detection",
        "clinical decision support systems"
    ]

    expanded = []

    for keyword in base_keywords:
        keyword_embedding = semantic_model.encode(keyword, convert_to_tensor=True)
        bank_embeddings = semantic_model.encode(concept_bank, convert_to_tensor=True)

        similarities = util.cos_sim(keyword_embedding, bank_embeddings)[0]

        for i, score in enumerate(similarities):
            if score > 0.5:  # similarity threshold
                expanded.append(concept_bank[i])

    return list(set(expanded))




app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

CORS(app)

kw_model = KeyBERT()
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route("/keywords", methods=["POST"])
def generate_keywords():
    data = request.json
    text = data.get("text", "")

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=8
    )

    cleaned_keywords = clean_keywords(keywords)

    expanded_keywords = expand_keywords(cleaned_keywords)

    all_keywords = list(set(cleaned_keywords + expanded_keywords))

    return jsonify({"keywords": all_keywords})

if __name__ == "__main__":
    app.run(debug=True)
