
import pandas as pd

from config import RAW_DATA_PATH


def load_raw_data():

    """
    Load the original motorcycle mileage dataset.
    """

    data = pd.read_csv(
        RAW_DATA_PATH,
        low_memory=False
    )

    return data
