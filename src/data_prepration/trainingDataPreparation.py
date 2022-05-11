import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from database import Database

# this file can be used to convert the data to ratio rather than absolute mass format.

# Loading original dataset_type
data = Database()


def convert_to_ratio(df, save_dir = "data/original", name="PredictedData_1.csv"):
    """converting the absolute of mass of each component to a ratio

    Args:
        df (dataframe): data for conversion
        save_dir (str, optional): save directory. Defaults to "data/original".
        name (str, optional): file name. Defaults to "PredictedData_1.csv".
    """
    df = df.assign(PcEcPc=lambda x: df['PC'] / (df['EC'] + df['PC']))
    df = df.assign(liEcPc=lambda x: df['LiPF6'] / (df['EC'] + df['PC']))
    columns = ["experimentID", "PcEcPc", "liEcPc", "temperature", "conductivity", "resistance"]
    df[columns].to_csv(os.path.join(save_dir, name))

convert_to_ratio(data._raw_initial_data, name="OriginalData.csv")