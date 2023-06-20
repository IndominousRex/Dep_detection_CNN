import torch
from torch import nn


# time distributed class to mimic keras timedistributed layer (going through every temporal slice of the input)
# it reshapes input dimensions to pass through layer and then changes them back again
class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.view([-1] + list(x.shape[2:]))

        y = self.module(x_reshape).squeeze()

        y = y.view(list(x.shape[:2]) + list(y.shape[1:]))

        return y


# main CNN
class CNN1(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, max_len, max_posts, num_class=2, kernel_size=3
    ):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.td1 = TimeDistributed(self.emb)

        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=max_len, out_channels=25, kernel_size=kernel_size, stride=1
            ),
            nn.ReLU(),
            nn.AvgPool1d((embed_dim - kernel_size + 1)),
        )

        self.td2 = TimeDistributed(self.conv_block_1)

        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=max_posts, out_channels=26, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=26 * 25, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=num_class),
            nn.Softmax(),
        )
        self.lin = nn.Linear(embed_dim, num_class)

    def forward(self, x: torch.Tensor):
        return self.dense_block(self.conv_block_2(self.td2(self.td1(x))))
