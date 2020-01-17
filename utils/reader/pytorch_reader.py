from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader
from pathlib import Path


def get_data_loader(
        data_path: str = None,
        num_epochs: int = 1,
        batch_size: int = 16
):
    if not data_path:
        return None

    return DataLoader(
        make_batch_reader(
            dataset_url=data_path,
            num_epochs=num_epochs
        ),
        batch_size=batch_size
    )
