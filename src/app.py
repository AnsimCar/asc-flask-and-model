# app.py
from flask import Flask, request, render_template
import os
import Service
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/convert/<path:img>')
def convert(img):
  img_path = Service.checkCar(img)
  return img_path

@app.route('/test')
def testConvert():
  Service.fileOpen()

if __name__=="__main__":
  app.run(debug=True)
  # app.run(host="127.0.0.1", port="5000", debug=True)