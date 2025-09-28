import torch.cuda
from models import SimpleNeuralNetwork
from dataset import AnimalDataset
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from sklearn.metrics import classification_report, accuracy_score
from torch import argmax
from tqdm import tqdm


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    root = "../Animals"
    transform = Compose([
        Resize((224,224)),
        ToTensor()
    ])
    training_data = AnimalDataset(root = root, train = True, transform= transform)
    test_data = AnimalDataset(root = root, train  = False, transform= transform)

    train_dataloader = DataLoader(
        dataset = training_data,
        batch_size= 8,
        drop_last= True,
        shuffle= True,
        num_workers= 4
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=8,
        drop_last=True,
        shuffle=False,
        num_workers=4
    )

    num_epoch = 100
    num_iter = len(train_dataloader)
    model = SimpleNeuralNetwork().to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 1e-3, momentum= 0.9)

    for epoch in range(num_epoch):
        progress_bar = tqdm(train_dataloader, colour= "green")
        model.train()
        for iter, (image, label) in enumerate(progress_bar):
            if torch.cuda.is_available():
                image = image.to(device)
                label = label.to(device)

            # forward
            output = model(image)
            loss_value = criterion(output, label)

            #backward

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            progress_bar.set_description("Epoch {}/{} Loss_values {:.3f} Iteration {}/{}".format(epoch + 1,num_epoch, loss_value, iter + 1, num_iter ))


        model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for image, label in train_dataloader:
                all_labels.extend(label)
                if torch.cuda.is_available():
                    image = image.to(device)
                    label = label.to(device)

                output = model(image)
                loss_value = criterion(output, label)
                indices = argmax(output, dim= 1)
                all_predictions.extend(indices)
            all_predictions = [prediction.item() for prediction in all_predictions]
            all_labels = [label.item() for label in all_labels]
            print(all_predictions)
            print("------------------------")
            print(all_labels)

            accuracy = accuracy_score(all_labels, all_predictions)
            print("Acurracy : {}".format(accuracy))
            report = classification_report (all_labels, all_predictions)
            print(report)






