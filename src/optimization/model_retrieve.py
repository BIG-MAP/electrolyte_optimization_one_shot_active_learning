# This file can be used for retreiving the saved trained model and use it for calculating the output of any point of interest.

import os
import sys
import joblib
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from utils import *
from database import Database
from utils import *
from ml_util import *

# Loading original dataset_type
data = Database()
df = data._initial_data
temperatures = data.sorted_temperatures
predict_data = []
temperature_order = []
blind_test = pd.DataFrame()
SAVE = False
path_to_model = os.path.join("data", "augmented", "regression", "separate_datasets", "poly_before_one_shot", "old")

meshgrid_test_x = [0.098316619, 0.098369507,0.098514549,0.098538967,0.098742297,0.098742364,0.098753423,0.098821919,0.098973291,0.09923281,0.10005633,0.10013993,0.100944341,0.109512134,0.110279441,0.110478765,0.323299518,0.327887589,0.330359239,0.336842455,0.33722314,0.344755564,0.346060962, 0.346792576]
meshgrid_test_y = [0.87224212,0.301443329,0.252490738,0.786373263,0.274568453,0.85447359,0.797269059,0.282487301,0.290565767,0.865168137,0.797563723,0.823472427,0.83212906,0.292150801,0.271920445,0.262004695,0.280226177,0.300625252,0.273001711,0.272217056,0.301752066,0.29358397,0.284250375,0.287756431]


for temp in temperatures:
    print(temp)
    loaded_model = joblib.load(open(fr"{path_to_model}\poly_model_{temp}.pkl", "rb"))

    temp_col = [temp for i in range(len(meshgrid_test_x))]
    x_test_grid = pd.DataFrame(dict(PcEcPc = meshgrid_test_x, liEcPc = meshgrid_test_y, 
                                    temperature = temp_col))
    pred = loaded_model.predict(x_test_grid)
    predict_data.extend(pred)
    temperature_order.extend(temp_col)

# append the predicted value to the dataframe containing the real measurement
df_retrieve = pd.DataFrame(predict_data, columns=["exact_prediction"])
df_temp = pd.DataFrame(temperature_order, columns=["temp_check"])

average_data = pd.read_csv(os.path.join("data","formulation_suggestions", "comparison_experiment_2.csv"))
comparison_data = pd.concat([average_data, df_retrieve, df_temp], axis=1)

if SAVE:
    comparison_data.to_csv(os.path.join("data","formulation_suggestions", "comparison_experiment_finale.csv"))
