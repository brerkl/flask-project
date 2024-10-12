from flask import Flask, render_template, request

from PIL import Image
import pickle

import numpy as np

app = Flask(__name__)
with open('flask/model.pkl', 'rb') as file:
    model = pickle.load(file)

mlb = np.array(['quả Chuối', 'Trái dưa leo', 'Tốt', 'Quả nho', 'Vẫn có thể ăn được', 'Hư thối']).astype('object')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    labels, probs = [], []
    
    imagefile = request.files['imagefile']
    image_path = "./flask/assets/images/" + imagefile.filename
    imagefile.save(image_path)
    
    # Tiền xử lý dữ liệu
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.array(image, dtype="float") / 255.0
    image = image.astype('float32')
    # Dự đoán
    result = model.predict(np.expand_dims(image, axis=0))[0]
    # Trích xuất 2 dự đoán có tỷ lệ cao nhất 
    argmax = np.argsort(result)[::-1][:2]
    labels = [mlb[j] for _, j in enumerate(argmax)]
    classification = f'{labels[0]} với chất lượng {labels[1]}'

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=5000, debug=True)