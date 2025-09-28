import torch
import cv2
import numpy  as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from models import SimpleConvolutionNeuralNetwork
from torch.nn import Softmax
from fastapi.middleware.cors import CORSMiddleware
import time
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from torchvision.transforms import Compose, Resize, ToTensor
image_size = 224
categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = SimpleConvolutionNeuralNetwork().to(device)
checkpoint = torch.load("./trained_model/best_cnn.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
softmax = Softmax(dim = 1)
model.eval()

@app.post("/predict")
async def predict(file :UploadFile = File(...)):
    start_time = time.time()
    content = await file.read()


    nparr = np.frombuffer(content, np.uint8)
    org_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # tensor,dung kenh, 4 kenh mau , cung kich thuoc
    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    image = np.transpose(image, (2, 0, 1))/255.
    image = torch.from_numpy(image).float().unsqueeze(0).to(device)




    with torch.no_grad():
        output = model(image)
        probs = softmax(output)
        prediction = torch.argmax(probs)
        label = categories[prediction.item()]

    end_time = time.time()
    return {
        "class": label,
        "confidence": float(probs[0][prediction].item()),
        "processing_time": end_time - start_time
    }
