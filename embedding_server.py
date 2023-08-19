from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)  # You can use any other model you prefer

app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello World"

@app.route('/transform', methods=['POST'])
def transform_sentences():
    try:
        data = request.json
        sentences = data['sentences']
        
        if not sentences or not isinstance(sentences, list):
            return jsonify({'error': 'Invalid input format'}), 400
        
        embeddings = model.encode(sentences)
        return jsonify({'embeddings': embeddings.tolist()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
