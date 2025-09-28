import cv2
import torch.cuda
from models import SimpleConvolutionNeuralNetwork
from dataset import AnimalDataset
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch import argmax
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np



#train mô hình
# sử dụng parser để truyển tham số
#writer để lưu kết quả
# tạo checkponint lưu epoch, model, optimize
# lưu confusion matrix
# data augmentation

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--root", "-r", type = str, default= "../Animals")
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default= 8)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_model", "-t", type=str, default="trained_model")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)

    return parser.parse_args()

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


if __name__ == '__main__':
    args = get_args()

    if not os.path.isdir(args.logging):
        os.makedirs(args.logging)
    # else:
    #     shutil.rmtree(args.logging)

    if not os.path.isdir(args.trained_model):
        os.makedirs(args.trained_model)



    # using GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Loading data
    root = args.root
    train_transform = Compose([
        Resize((args.image_size,args.image_size)),
        ToTensor(),
        RandomAffine(degrees=[-90, 90], translate=None, scale=None,
                     shear=None,
                     fill=0, center=None),
        ColorJitter(brightness=[1, 1], contrast=[1, 1],
                    saturation=[1, 1], hue=[0, 0])

    ])

    test_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),

    ])
    training_data = AnimalDataset(root = root, train = True, transform= train_transform)
    test_data = AnimalDataset(root = root, train  = False, transform= test_transform)

    train_dataloader = DataLoader(
        dataset = training_data,
        batch_size= args.batch_size,
        drop_last= True,
        shuffle= True,
        num_workers= 4
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size= args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=4
    )

    # Define num_epoch, model, optimizer, criterion
    num_epoch = args.epoch
    num_iter = len(train_dataloader)
    model = SimpleConvolutionNeuralNetwork().to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 1e-3, momentum= 0.9)
    writer = SummaryWriter(args.logging)
    best_acc = 0

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0


    # Loop through epochs
    for epoch in range( start_epoch,num_epoch):
        progress_bar = tqdm(train_dataloader, colour= "green")

        # Training phase
        model.train()
        for iter, (image, label) in enumerate(progress_bar):
            if torch.cuda.is_available():
                image = image.to(device)
                label = label.to(device)

            # forward
            output = model(image)
            loss_value = criterion(output, label)
            progress_bar.set_description("Epoch {}/{} Loss_values {:.3f} Iteration {}/{}".format(epoch + 1,num_epoch, loss_value, iter + 1, num_iter ))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iter + iter + 1)

            #backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for iter, (image, label) in enumerate(test_dataloader):
                all_labels.extend(label)
                if torch.cuda.is_available():
                    image = image.to(device)
                    label = label.to(device)

                output = model(image)
                loss_value = criterion(output, label)
                indices = argmax(output, dim= 1)
                all_predictions.extend(indices)


            # Store values and evaluate accuracy
            all_predictions = [prediction.item() for prediction in all_predictions]
            all_labels = [label.item() for label in all_labels]

            checkpoint = {
                "model" : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch" : epoch + 1,
                "best_acc": best_acc
            }

            torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_model))

            accuracy = accuracy_score(all_labels, all_predictions)
            print("Accuracy : {}".format(accuracy))
            report = classification_report (all_labels, all_predictions)
            print(report)
            writer.add_scalar("Val/Accuracy", accuracy, epoch )
            plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions),training_data.categories, epoch)

            #store best accuracy
            if accuracy > best_acc:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_acc" : best_acc
                }
                torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_model))
                best_acc = accuracy










