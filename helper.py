import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import tqdm
import numpy as np


def run(mode, dataloader, model, optimizer=None, use_cuda=True, class_weights=None):
    """
    mode: either "train" or "valid". If the mode is train, we will optimize the model
    class_weights: optional tensor of weights for each class to handle imbalance
    """
    running_loss = []
    
    # Use weighted loss if provided
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    actual_labels = []
    predictions = []
    for inputs, labels in tqdm.tqdm(dataloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss.append(loss.item())

        actual_labels += labels.view(-1).cpu().numpy().tolist()
        _, pred = torch.max(outputs, dim=1)

        predictions += pred.view(-1).cpu().numpy().tolist()

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc = np.sum(np.array(actual_labels) == np.array(
        predictions)) / len(actual_labels)
    print(mode, "Accuracy:", acc)

    loss = np.mean(running_loss)

    return loss, acc