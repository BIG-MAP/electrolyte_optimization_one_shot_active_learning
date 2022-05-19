# This file can be used for making a dataframe from measured conductivity for new formulations
import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

import pandas as pd
from utils import *
from database import Database

nearest_idx = [7700, 2000, 1500, 6800, 1700, 7500, 6900, 1800, 1900, 7600, 6901,
               7200, 7300, 1901, 1702, 1602, 1825, 2025, 1726, 1726, 2026, 1927, 1827, 1928]

# Loading original dataset_type
data = Database()
df = data._initial_data
temperatures = data.sorted_temperatures
generator = "poly"


retrieve_data = []
blind_test = pd.DataFrame()
for temp in temperatures:
    regressor_name = f"{generator}_{float(temp)}.csv"
    data = load_dataset(dataset_type="augmented", method="regression", aggregation=False,
                        generator=generator, name=regressor_name)
    for idx in nearest_idx: 
        retrieve_data.append([data["PcEcPc"].loc[[idx]].values[0], 
                              data["liEcPc"].loc[[idx]].values[0],data["liEcPc"].loc[[idx-100]].values[0],
                              data["liEcPc"].loc[[idx+100]].values[0], data["conductivity"].loc[[idx]].values[0]
                              ,data["conductivity"].loc[[idx-100]].values[0],data["conductivity"].loc[[idx+100]].values[0],
                              data["temperature"].loc[[idx]].values[0]])        


df_retrieve = pd.DataFrame(retrieve_data, columns=["nearest_PcEcPc", "nearest_liEcPc",
                                                   "nearest_liEcPc_min","nearest_liEcPc_max", 
                                                   "nearest_conductivity", "nearest_conductivity_min",
                                                   "nearest_conductivity_max","nearest_temperature"])

average_data = pd.read_csv(os.path.join("data","formulation_suggestions", "average_experiment_with_res.csv"))
comparison_data = pd.concat([average_data, df_retrieve], axis=1)
comparison_data.to_csv(os.path.join("data","formulation_suggestions", "comparison_experiment_finale.csv"))
