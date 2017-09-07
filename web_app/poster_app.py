import os
from flask import Flask, render_template, request, session
from PIL import Image
import numpy as np
from Metrics import f1,recall
from keras.models import load_model
import sys

model = sys.argv[1]

def process(img):
    return (np.array(Image.open(img).resize((200,300),Image.NEAREST))/255.).reshape(1,300,200,3)

def prediction(im):
    model = load_model(model,custom_objects={'recall':recall,'f1':f1})
    genre = ['Horror','Romance','Adventure','Documentary']
    preds = model.predict(im)
    probs = np.round(preds*100.0,2)
    return probs

app = Flask(__name__,static_folder='images')
app.secret_key = 'abcd'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
#    session['model'] = load_m()
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT,'images/')

    if not os.path.isdir(target):
        os.mkdir(target)

    f = request.files.get('file')
    filename = f.filename
    destination = '/'.join([target,filename])
    f.save(destination)
    session['img'] = filename
    return render_template('complete.html',image_name=filename)

@app.route('/predict', methods=['POST'])
def predict():
    img = session.get('img',None)
    im = process('images/'+img)
    result = prediction(im)
    return render_template('predict.html',pred=result,image_name=img)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
