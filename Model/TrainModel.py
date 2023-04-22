import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(y_true, y_pred):
    return torch.tensor(torch.sum(y_true==y_pred).item()/len(y_pred))

def train(model, train_dl, valid_dl, learning_rate = 0.01, epoch = 10):
    """Train model

    Parameters
    ----------
    model : object
        model that have to train
    train_dl : dataloader
        train dataloader (images, labels)
    valid_dl : dataloader
        validation dataloader (images, labels)
    learning_rate : float, optional
        learning rate, by default 0.01
    epoch : int, optional
        number of time to train, by default 10

    Returns
    -------
    tuple (list, list, list, list)
        contain training loss, training accuracy, validation loss and validation accuracy
    """
    optimizer = Adam(model.parameters(), learning_rate = learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    for i in range(epoch):
        train_loss = []
        val_loss = []
        train_accuracy = []
        val_accuracy = []
        train_loop = tqdm(train_dl, leave=True)
        for x, labels in train_loop:
            train_loop.set_description(f"Epoch {i+1}")
            optimizer.zero_grad()
            y = model(x)
            y = F.softmax(y)
            loss = loss_fn(y.float(), labels)
            loss.backward()
            optimizer.step()

            _, pred =torch.max(y, dim = 1)
            accuracy_val = accuracy(labels, pred)
            train_loss.append(loss.item())
            train_accuracy.append(accuracy_val.item())

            train_loop.set_postfix(
                train_loss=sum(train_loss) / len(train_loss),
                train_accuracy=sum(train_accuracy) / len(train_accuracy),
            )

        val_loop = tqdm(valid_dl, leave=True)
        with torch.no_grad():
            for x, labels in val_loop:
                y = model(x)
                y = F.softmax(y)
                loss = loss_fn(y.float(), labels)

                _, pred =torch.max(y, dim = 1)
                accuracy_val = accuracy(labels, pred)
                val_loss.append(loss.item())
                val_accuracy.append(accuracy_val.item())

                val_loop.set_postfix(
                    train_loss=sum(train_loss) / len(train_loss),
                    train_accuracy=sum(train_accuracy) / len(train_accuracy),
                    val_loss=sum(val_loss) / len(val_loss),
                    val_accuracy=sum(val_accuracy) / len(val_accuracy),
                )

        training_loss.append(sum(train_loss) / len(train_loss))
        training_accuracy.append(sum(train_accuracy) / len(train_accuracy))
        validation_loss.append(sum(val_loss) / len(val_loss))
        validation_accuracy.append(sum(val_accuracy) / len(val_accuracy))
    
    return (
        training_loss,
        training_accuracy,
        validation_loss,
        validation_accuracy
    )

def save_model(model,path):
    torch.save(model, path)
