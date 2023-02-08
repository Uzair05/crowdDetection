import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import cv2
import torch


model = torch.hub.load("./yolov5", "custom", source='local', path="./modelSaves/best.pt", force_reload=True)

def predictPerson(file):
    res = model(file)
    res = pd.DataFrame(res.xywh[0].cpu().numpy(), columns=["x", "y", "w", "h", "confidence", "label"])
    return res.to_json(orient="index")
