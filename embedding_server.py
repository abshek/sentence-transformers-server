from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import base64
import uuid
from PIL import Image
import os

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
        
        if type == "image":
            filename='images/'+str(uuid.uuid4())+'.jpg'
            #decode base64 string data
            decoded_data=base64.b64decode((input))
            #write the decoded data back to original format in  file
            img_file = open(filename, 'wb')
            img_file.write(decoded_data)
            img_file.close()
            embeddings = img_model.encode(Image.open(filename))
            if os.path.exists(filename):
                os.remove(filename)
            else:
                print("The file does not exist") 
        else:
            embeddings = txt_model.encode(input)

        return jsonify({'embeddings': embeddings.tolist()[0]}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
