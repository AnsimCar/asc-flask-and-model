# app.py
from flask import Flask
import Service

app = Flask(__name__)

@app.route('/convert/<email>/<path:img>')
def convert(email, img):
  return Service.checkCar(email, img) 

if __name__=="__main__":
  app.run(debug=True)
  # app.run(host="127.0.0.1", port="5000", debug=True)