from argparse import ArgumentParser
from models import SimpleConvolutionNeuralNetwork
import torch
import cv2
import numpy as np
from torch.nn import Softmax
from torchvision.transforms import ToTensor, Resize


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_file", "-if", type = str, default= "../Animals/test_images/3.jpg")
    parser.add_argument("--checkpoint", "-c", type = str, default= "../Animals/trained_model/best_cnn.pt")
    parser.add_argument("--image_size", "-i", type = int, default= 224)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SimpleConvolutionNeuralNetwork().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("no training found")
        exit()

    categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    org_image = cv2.imread(args.image_file)
    #tensor,dung kenh, 4 kenh mau , cung kich thuoc
    image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB )
    image = cv2.resize(image, (args.image_size,args.image_size))
    image = np.transpose(image, (2,0,1))/255.
    image = image[None,:,:,:]
    image = torch.from_numpy(image).float().to(device)

    softmax = Softmax(dim = 1)
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = softmax(output)
        indexes = torch.argmax(probs)
        prediction = categories[indexes.item()]


    cv2.imshow("{}".format(prediction), org_image)
    cv2.waitKey(0)
