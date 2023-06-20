import torch
from torch import nn
from torchinfo import summary
import pickle
import warnings

warnings.filterwarnings("ignore")

from data_setup import get_dataloaders
from model import CNN1
from engine import train
from utils import save_model

# set hyperparameters
LEARNING_RATE = 0.001
MAX_LENGTH = 150
BATCH_SIZE = 16
MODE = "earliest"
NUM_EPOCHS = 10
MAX_POSTS = 400 if MODE == "earliest" else 1500

# setting target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# setting seed
torch.cuda.manual_seed(42)
torch.manual_seed(42)

# cereating dataloaders
train_dataloader, vocab_len = get_dataloaders(
    type="training", max_length=MAX_LENGTH, batch_size=BATCH_SIZE, mode=MODE
)

"""test_dataloader = get_dataloaders(
    type="testing", max_length=MAX_LENGTH, batch_size=BATCH_SIZE, mode=MODE
)"""

# creating model using model.py file
model = CNN1(
    vocab_size=vocab_len, embed_dim=50, max_len=MAX_LENGTH, max_posts=MAX_POSTS
)

# setting loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# printing model summary
print(
    summary(
        model=model,
        input_data=torch.rand(size=(16, 400, 150)).type(torch.int64),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
)

# training and testing the model
results = train(
    model=model,
    train_dataloader=train_dataloader,
    # test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)

pickle.dump(results, open("model_results.pkl", "wb"), protocol=-1)

# saving the model
save_model(
    model=model, target_dir="models", model_name=f"model_{MAX_POSTS}_{MAX_LENGTH}.pth"
)
