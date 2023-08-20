from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

text_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
img_model_name = 'clip-ViT-B-32'
txt_model = SentenceTransformer(text_model_name)
img_model = SentenceTransformer(img_model_name)

app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello World"

@app.route('/transform', methods=['POST'])
def transform_sentences():
    try:
        data = request.json
        input = data['input']
        type = data['type']
        
        if not input or not isinstance(input, list):
            return jsonify({'error': 'Invalid input format'}), 400
        
        model
        
        if type == "image":
            model = txt_model
        else:
            model = img_model
        embeddings = model.encode(input, convert_to_tensor=True)
        return jsonify({'embeddings': embeddings.tolist()[0]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
