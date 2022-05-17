import sys
import os.path
from matplotlib.pyplot import colorbar
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from utils import *
from database import Database
import pandas as pd

## This file can be used for plotting the predicted conductivity for various formulation while considering the real measurements
## This file is used when we do not want to train anything and just want to modify the plots.
 
# Loading original dataset_type
data = Database()
df = data._initial_data
new_df = data._updated_data
temperatures = data.sorted_temperatures
REAL = True
generators = ["poly"]  # "ensamble", "tvae
cols = ["PcEcPc", "liEcPc", "temperature", "conductivity", "kind"]
data_types=["old", "new"]


for data_type in data_types:
    for generator in generators:
        top_result = pd.DataFrame()
        for temp in temperatures:
            regressor_name = f"{generator}_{float(temp)}.csv"
            data = load_dataset(dataset_type="augmented", method="regression", aggregation=False,
                                generator=generator, name=regressor_name, data_type=data_type)
            if REAL:
                real_data = True
                if data_type=="old":
                    x_real, y_real = df["PcEcPc"], df["liEcPc"]
                else:
                    x_real, y_real = new_df["PcEcPc"], new_df["liEcPc"]
            else:
                x_real, y_real = None, None

            plot_material(x=data["PcEcPc"], y=data["liEcPc"], colour=data["conductivity"], x_label = r"$\frac{PC}{EC+PC}$", y_label=r"$\frac{LiPF_{6}}{Ec + PC}$",
                        file_name=f"{generator}_{temp}.png", dataset_type="augmented", generator=generator,figsize=(4,4),
                        contour=True, square=True, colorbar=True, retrain=True, contour_level=20, real_data=real_data,
                        x_real=x_real, y_real=y_real, data_type=data_type, uncertaiinty=False, ROTATION=True)  