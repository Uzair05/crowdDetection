import json
from typing import *
from flask import Flask, request
import cv2
import numpy as np
from model import predictPerson


app = Flask(__name__)


@app.route('/crowd_details', methods=['POST'])
def crowd_details():
    # request.args.get("api_token", None, type=str)
    file_ = request.files['file']
    file_name_ = file_.filename
    file__ = cv2.imdecode(np.fromfile(file_, np.uint8), cv2.IMREAD_COLOR)
    return {"file_name": file_name_}


app.run(host='0.0.0.0')
