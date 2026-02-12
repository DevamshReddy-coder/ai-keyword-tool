from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT
from flask import render_template
from sentence_transformers import SentenceTransformer, util
import re
import requests




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

def refine_keywords(keywords):
    refined = []

    # sort by length (longer phrases first)
    keywords = sorted(keywords, key=len, reverse=True)

    for kw in keywords:
        if not any(kw in other and kw != other for other in refined):
            refined.append(kw)

    return refined

def cluster_keywords(keywords):
    clusters = {
        "problem_terms": [],
        "method_terms": [],
        "data_terms": [],
        "other_terms": []
    }

    method_words = ["learning", "network", "model", "algorithm", "cnn", "ai"]
    data_words = ["image", "text", "signal", "speech", "mri", "xray"]

    for kw in keywords:
        if any(word in kw for word in method_words):
            clusters["method_terms"].append(kw)
        elif any(word in kw for word in data_words):
            clusters["data_terms"].append(kw)
        elif len(kw.split()) >= 2:
            clusters["problem_terms"].append(kw)
        else:
            clusters["other_terms"].append(kw)

    return clusters

def build_boolean_query(or_groups, not_group=None):
    query_parts = []

    # Handle OR groups connected by AND
    for group in or_groups:
        if group:
            or_part = " OR ".join([f'"{term}"' for term in group])
            query_parts.append(f"({or_part})")

    # Join all OR groups using AND
    final_query = " AND ".join(query_parts)

    # Handle NOT group
    if not_group:
        not_part = " OR ".join([f'"{term}"' for term in not_group])
        final_query += f" NOT ({not_part})"

    return final_query


def fetch_dynamic_titles(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 20,
        "fields": "title"
    }

    response = requests.get(url, params=params)

    titles = []

    if response.status_code == 200:
        data = response.json()
        for paper in data.get("data", []):
            if paper.get("title"):
                titles.append(paper["title"])

    return titles





def build_concept_bank():
    concepts = set()

    with open("research_titles.txt", "r", encoding="utf-8") as f:
        titles = f.readlines()

    for title in titles:
        title = title.lower()

        # remove special characters
        title = re.sub(r"[^a-z0-9\s]", "", title)

        words = title.split()

        # create 2-word and 3-word phrases
        for i in range(len(words)):
            if i + 1 < len(words):
                phrase2 = words[i] + " " + words[i+1]
                concepts.add(phrase2)
            if i + 2 < len(words):
                phrase3 = words[i] + " " + words[i+1] + " " + words[i+2]
                concepts.add(phrase3)

    return list(concepts)


def expand_keywords(base_keywords):

    titles = fetch_dynamic_titles(" ".join(base_keywords))

    concept_bank = []

    # Extract phrases from titles
    for title in titles:
        title = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower())
        words = title.split()

        for i in range(len(words)):
            if i + 1 < len(words):
                concept_bank.append(words[i] + " " + words[i+1])
            if i + 2 < len(words):
                concept_bank.append(words[i] + " " + words[i+1] + " " + words[i+2])

    expanded = []

    # Compare semantic similarity
    for keyword in base_keywords:
        keyword_embedding = semantic_model.encode(keyword, convert_to_tensor=True)
        bank_embeddings = semantic_model.encode(concept_bank, convert_to_tensor=True)

        similarities = util.cos_sim(keyword_embedding, bank_embeddings)[0]

        for i, score in enumerate(similarities):
            if score > 0.65:
                expanded.append(concept_bank[i])

    return list(set(expanded))


def fetch_papers(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,year,authors"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        papers = []

        for paper in data.get("data", []):
            papers.append({
                "title": paper.get("title"),
                "year": paper.get("year"),
                "authors": ", ".join([a["name"] for a in paper.get("authors", [])])
            })

        return papers
    else:
        return []





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
    final_keywords = refine_keywords(all_keywords)

    clusters = cluster_keywords(final_keywords)

    return jsonify({
    "clusters": clusters,
    "keywords": final_keywords
})

@app.route("/build-query", methods=["POST"])
def build_query_route():
    data = request.json

    or_groups = data.get("or_groups", [])
    not_group = data.get("not_group", [])

    boolean_query = build_boolean_query(or_groups, not_group)

    return jsonify({
        "boolean_query": boolean_query
    })






if __name__ == "__main__":
    app.run(debug=True)

