# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms

os.chdir(
    "C:/Users/annas/OneDrive/Dokumente/DTU/8_semester/3weeks_MLops/MLops_exercises_organized"
)
print(os.getcwd())
DATA_PATH = "data\\processed\\"


@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():  # (input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # Define a transform to normalize the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Download and load the training data
    trainset = datasets.MNIST(DATA_PATH, download=True, train=True, transform=transform)
    main()
