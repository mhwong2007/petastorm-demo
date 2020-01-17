from pathlib import Path

import torch

from model.linear import get_linear_model
from utils.reader.pytorch_reader import get_data_loader
from sklearn.metrics import classification_report


def train_model(
        train_path: str,
        device: str = 'cpu',
        num_epoch: int = 50
):
    # get model
    model = get_linear_model()

    # define loss function
    loss_fun = torch.nn.CrossEntropyLoss()

    # define optimiser
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(num_epoch):
        batch_losses = 0
        for batch, data in enumerate(get_data_loader(train_path)):
            opt.zero_grad()

            features = data['features'].to(device)
            label = data['label'].to(device)

            y_pred = model(features)
            loss = loss_fun(y_pred, label)

            loss.backward()
            opt.step()

            # track batch losses
            batch_losses += loss.item()

        avg_batch_loss = batch_losses / (batch + 1)
        print(f'Epoch {epoch} average loss: {avg_batch_loss}')

    return model


def evaluation(model, data_path: str):
    true_label_list = []
    pred_label_list = []

    model.eval()

    for data in get_data_loader(data_path):
        features = data['features'].to('cpu')
        label = data['label'].to('cpu').tolist()

        y_pred = model(features)
        _, label_pred = torch.max(y_pred, 1)

        label_pred = label_pred.tolist()

        true_label_list.extend(label)
        pred_label_list.extend(label_pred)

    print(
        classification_report(
            true_label_list,
            pred_label_list,
            target_names=[
                'Iris-setosa',
                'Iris-versicolor',
                'Iris-virginica'
            ]
        )
    )


def train_job(
        train_path: str = None,
        test_path: str = None,
        model_output_path: str = None
):
    if not train_path:
        train_path = Path('data/spark_processed_data/train_iris.parquet').absolute().as_uri()
    if not test_path:
        test_path = Path('data/spark_processed_data/test_iris.parquet').absolute().as_uri()
    if not model_output_path:
        # mkdir
        model_dir_path = Path('output_model')
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_output_path = str((model_dir_path / 'iris.torch.model').absolute())

    # train model
    model = train_model(train_path)

    # evaluation
    print('Train set performance:')
    evaluation(model, train_path)

    print('Test set performance')
    evaluation(model, test_path)

    # save model
    torch.save(model.state_dict(), model_output_path)
    print(f'Model saved at {model_output_path}')
