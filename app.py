from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
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


import time

def fetch_dynamic_titles(query):

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 20,
        "fields": "title"
    }

    for attempt in range(2):  # try twice
        try:
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                titles = []

                for paper in data.get("data", []):
                    if paper.get("title"):
                        titles.append(paper["title"])

                if titles:
                    return titles

        except Exception as e:
            print("API attempt failed:", e)

        time.sleep(1)  # wait 1 second before retry

    print("Semantic Scholar returned nothing. Trying OpenAlex...")
    openalex_titles = fetch_openalex_titles(query)

    print("OpenAlex titles fetched:", len(openalex_titles))

    return openalex_titles


def fetch_openalex_titles(query):

    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": 20
    }

    try:
        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            titles = []

            for work in data.get("results", []):
                if work.get("title"):
                    titles.append(work["title"])

            return titles

    except Exception as e:
        print("OpenAlex failed:", e)

    return []





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



def expand_keywords(base_keywords, max_depth=2, top_k=8):
    all_expanded = set(base_keywords)

    current_level = base_keywords

    for depth in range(max_depth):
        print(f"Expansion depth: {depth+1}")

        titles = fetch_dynamic_titles(" ".join(current_level))
        print("Number of titles fetched:", len(titles))

        if not titles:
            break

        concept_bank = []

        for title in titles:
            title = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower())
            words = title.split()

            for i in range(len(words)):
                if i + 1 < len(words):
                    concept_bank.append(words[i] + " " + words[i+1])
                if i + 2 < len(words):
                    concept_bank.append(words[i] + " " + words[i+1] + " " + words[i+2])

        concept_bank = list(set(concept_bank))

        if not concept_bank:
            break

        keyword_embeddings = semantic_model.encode(current_level, convert_to_tensor=True)
        bank_embeddings = semantic_model.encode(concept_bank, convert_to_tensor=True)

        similarities = util.cos_sim(keyword_embeddings, bank_embeddings)

        new_terms = []

        for i in range(len(current_level)):
            for j in range(len(concept_bank)):
                if similarities[i][j] > 0.7:
                    new_terms.append(concept_bank[j])

        new_terms = list(set(new_terms))

        # Keep only top_k terms
        new_terms = new_terms[:top_k]

        if not new_terms:
            break

        all_expanded.update(new_terms)

        # Only expand strongest 3 next round
        current_level = new_terms[:3]

    return list(all_expanded)

def filter_phrases(phrases):
    bad_starts = {"of", "for", "and", "in", "on", "by", "with", "to"}
    bad_ends = {"of", "for", "and", "in", "on", "by", "with", "to"}

    cleaned = []

    for phrase in phrases:
        words = phrase.split()

        if len(words) < 2:
            continue

        if words[0] in bad_starts:
            continue

        if words[-1] in bad_ends:
            continue

        cleaned.append(phrase)

    return list(set(cleaned))

def rank_keywords_by_relevance(keywords, original_text):
    if not keywords:
        return keywords

    text_embedding = semantic_model.encode(original_text, convert_to_tensor=True)
    keyword_embeddings = semantic_model.encode(keywords, convert_to_tensor=True)

    similarities = util.cos_sim(text_embedding, keyword_embeddings)[0]

    scored = list(zip(keywords, similarities.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [kw for kw, score in scored]




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

socketio = SocketIO(app, cors_allowed_origins='*')

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

    expanded_keywords = filter_phrases(
    expand_keywords(cleaned_keywords)
)


    all_keywords = list(set(cleaned_keywords + expanded_keywords))
    final_keywords = refine_keywords(all_keywords)
    final_keywords = rank_keywords_by_relevance(final_keywords, text)


    clusters = cluster_keywords(final_keywords)
    # Emit suggestions + activity to connected realtime clients
    try:
        socketio.emit('suggestions', {'keywords': final_keywords, 'clusters': clusters})
        socketio.emit('activity', {'msg': f'Generated {len(final_keywords)} keywords'})
    except Exception:
        pass

    return jsonify({
        "clusters": clusters,
        "keywords": final_keywords
    })


@app.route("/expand-term", methods=["POST"])
def expand_term():
    data = request.json
    term = data.get("term", "").strip()

    if not term:
        return jsonify({"expanded": []})

    print(f"\nExpanding term: {term}")

    # Get research titles related ONLY to this term
    titles = fetch_dynamic_titles(term)

    # Safety check
    if not titles:
        print("No titles found for expansion.")
        return jsonify({"expanded": []})

    # Convert titles → phrases
    concept_bank = []
    for title in titles:
        title = re.sub(r"[^a-zA-Z0-9\s]", "", title.lower())
        words = title.split()

        for i in range(len(words)):
            if i + 1 < len(words):
                concept_bank.append(words[i] + " " + words[i+1])
            if i + 2 < len(words):
                concept_bank.append(words[i] + " " + words[i+1] + " " + words[i+2])

    # Semantic similarity filtering
    keyword_embedding = semantic_model.encode(term, convert_to_tensor=True)
    bank_embeddings = semantic_model.encode(concept_bank, convert_to_tensor=True)

    similarities = util.cos_sim(keyword_embedding, bank_embeddings)[0]

    expanded_terms = []
    for i, score in enumerate(similarities):
        if score > 0.55:   # lower than before → exploration mode
            expanded_terms.append(concept_bank[i])

    # Clean + refine
    expanded_terms = clean_keywords([(t, 1.0) for t in expanded_terms])
    expanded_terms = refine_keywords(expanded_terms)

    print("Expanded terms:", expanded_terms[:10])
    # Emit incremental suggestions and activity
    try:
        socketio.emit('suggestions_append', {'expanded': expanded_terms[:15]})
        socketio.emit('activity', {'msg': f"Expanded '{term}' → {len(expanded_terms[:15])} terms"})
    except Exception:
        pass

    return jsonify({
        "expanded": expanded_terms[:15]
    })


@app.route("/build-query", methods=["POST"])
def build_query_route():
    data = request.json

    or_groups = data.get("or_groups", [])
    not_group = data.get("not_group", [])

    boolean_query = build_boolean_query(or_groups, not_group)
    try:
        socketio.emit('activity', {'msg': 'Built boolean query'})
        socketio.emit('boolean_query', {'query': boolean_query})
    except Exception:
        pass

    return jsonify({
        "boolean_query": boolean_query
    })






if __name__ == "__main__":
    # Use SocketIO runner (eventlet recommended)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)

