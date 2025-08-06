from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sample stored questions
stored_questions = [
    "What is your name?",
    "How can I reset my password?",
    "Where is your office located?",
    "Tell me about your services.",
    "How do I make a payment?"
]

stored_embeddings = model.encode(stored_questions, convert_to_tensor=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    input_data = request.json
    input_text = input_data.get('text', '')
    input_embedding = model.encode(input_text, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(input_embedding, stored_embeddings)[0]

    best_idx = similarity_scores.argmax()
    best_score = similarity_scores[best_idx].item()
    best_match = stored_questions[best_idx]

    return jsonify({
        'input': input_text,
        'most_similar': best_match,
        'similarity_score': round(best_score, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
