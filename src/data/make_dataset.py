import logging
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms


def load_data(root_dir: str, output_filepath: str) -> None:
    """
    Generate and save train and test dataloader.

            Parameters:
                    root_dir (str): Location raw data
                    output_filepath (str): Location output files

            Returns:
                    None
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.FashionMNIST(
        root_dir + "/train", download=True, train=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        root_dir + "/test", download=True, train=False, transform=transform
    )

    logging.info(f"Training size: {len(trainset)}")
    logging.info(f"Test size: {len(testset)}")

    torch.save(trainset, f"{output_filepath}/train.pt")
    torch.save(testset, f"{output_filepath}/test.pt")


@click.command()
@click.argument(
    "input_filepath",
    default="data/raw",
    type=click.Path(exists=True),
)
@click.argument("output_filepath", default="data/processed/", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    load_data(input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
