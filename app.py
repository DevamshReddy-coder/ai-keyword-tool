from flask import Flask, request, jsonify
from flask_cors import CORS
from keybert import KeyBERT

app = Flask(__name__)
CORS(app)

kw_model = KeyBERT()

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

    keyword_list = [kw[0] for kw in keywords]
    return jsonify({"keywords": keyword_list})

if __name__ == "__main__":
    app.run(debug=True)
