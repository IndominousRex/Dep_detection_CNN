import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchmetrics.classification import Precision, Recall, F1Score


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    # Put model in train mode
    model.train()

    prec_fn = Precision(task="binary").to(device)
    rec_fn = Recall(task="binary").to(device)
    f1_fn = F1Score(task="binary").to(device)

    # Setup train loss and train accuracy values
    train_loss, train_acc, train_prec, train_recall, train_f1 = 0, 0, 0, 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculating and accumulating loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        train_prec += prec_fn(y_pred_class, y)
        train_recall += rec_fn(y_pred_class, y)
        train_f1 += f1_fn(y_pred_class, y)
        print(y_pred, y_pred_class, train_prec, "\n")

    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_prec /= len(dataloader)
    train_recall /= len(dataloader)
    train_f1 /= len(dataloader)
    return train_loss, train_acc, train_prec, train_recall, train_f1


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    # Put model in eval mode
    model.eval()

    prec_fn = Precision(task="binary").to(device)
    rec_fn = Recall(task="binary").to(device)
    f1_fn = F1Score(task="binary").to(device)

    # Setup test loss and test accuracy values
    test_loss, test_acc, test_prec, test_rec, test_f1 = 0, 0, 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
            test_prec += prec_fn(test_pred_labels, y)
            test_rec += rec_fn(test_pred_labels, y)
            test_f1 += f1_fn(test_pred_labels, y)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_prec /= len(dataloader)
    test_rec /= len(dataloader)
    test_f1 /= len(dataloader)

    return test_loss, test_acc, test_prec, test_rec, test_f1


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    # test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
) -> Dict[str, List]:
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
    }  # , "test_loss": [], "test_acc": []}

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        """test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )"""

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_prec: {train_acc:.4f} | "
            f"train_rec: {train_acc:.4f} | "
            f"train_f1: {train_acc:.4f} | "
            # f"test_loss: {test_loss:.4f} | "
            # f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)
        results["train_f1"].append(train_f1)
        # results["test_loss"].append(test_loss)
        # results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
