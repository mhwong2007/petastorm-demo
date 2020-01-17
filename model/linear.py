from torch import nn


def get_linear_model(
        input_dim: int = 4,
        output_dim: int = 3,
        device='cpu'
):
    net = nn.Sequential(
        nn.Linear(input_dim, 6),
        nn.ReLU(),
        nn.Linear(6, 4),
        nn.ReLU(),
        nn.Linear(4, output_dim)
    ).to(device)

    return net
