# app.py
from flask import Flask
import Service

app = Flask(__name__)

@app.route('/convert/<email>/<carId>/<sign>/<path:img>')
def convert(email, carId, sign, img):
  #sign 0-대여, 1-반납
  return Service.checkCar(email, carId, sign, img) 

if __name__=="__main__":
  app.run(debug=True)
  # app.run(host="127.0.0.1", port="5000", debug=True)