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

@app.route('/transform/text', methods=['POST'])
def transform_sentences():
    try:
        data = request.json
        sentences = data['sentences']
        
        if not sentences or not isinstance(sentences, list):
            return jsonify({'error': 'Invalid input format'}), 400
        
        embeddings = txt_model.encode(sentences, convert_to_tensor=True)
        return jsonify({'embeddings': embeddings.tolist()[0]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/transform/image', methods=['POST'])
def transform_images():
    try:
        data = request.json
        imgData = data['img']
        
        embeddings = img_model.encode(imgData, convert_to_tensor=True)
        return jsonify({'embeddings': embeddings.tolist()[0]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
