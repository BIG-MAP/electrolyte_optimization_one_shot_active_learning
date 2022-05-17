# This file can be used for post training
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import json
from tqdm import tqdm
import yaml 
from database import Database
from ml_util import *
import ml_util


REPEAT=True
np.random.seed(0)

data = Database()
old_df = data._initial_data
new_df = data._updated_data
trial_df = data._added_formulations
temperatures = data.sorted_temperatures

save_dir = os.path.join("data", "augmented", "regression")
df_test = load_dataset(dataset_type="augmented", method="regression", 
                       aggregation=False, name="poly_0.0.csv", generator="poly_before_one_shot")

meshgrid_test_x = [0.098316619, 0.098369507,0.098514549,0.098538967,0.098742297,0.098742364,0.098753423,0.098821919,0.098973291,0.09923281,0.10005633,0.10013993,0.100944341,0.109512134,0.110279441,0.110478765,0.323299518,0.327887589,0.330359239,0.336842455,0.33722314,0.344755564,0.346060962, 0.346792576]
meshgrid_test_y = [0.87224212,0.301443329,0.252490738,0.786373263,0.274568453,0.85447359,0.797269059,0.282487301,0.290565767,0.865168137,0.797563723,0.823472427,0.83212906,0.292150801,0.271920445,0.262004695,0.280226177,0.300625252,0.273001711,0.272217056,0.301752066,0.29358397,0.284250375,0.287756431]
X_test = df_test[["PcEcPc", "liEcPc"]]

data_types = ["new"]#, "old"] # 
fitting_functions = ["poly_fit", "bagging_extra_fit", "extratrees_fit"] 


all_results_for_plotting = {}

for generator in tqdm(fitting_functions):
    all_results_for_plotting[generator] = {}
    for data_type in data_types: 
        all_results_for_plotting[generator][data_type] = {}

        # save the best_params and best_result for comparison 
        best_params, best_result = pd.DataFrame(), pd.DataFrame()

        for temp in temperatures:
            all_results_for_plotting[generator][data_type][str(temp)] = {}
            if data_type == "new":
                TRAIN = True
                y_test_trial, x_test_trial = None, None

                print(f"temperature is {temp} and generator is {generator} and data type is {data_type}")
                df_temp = new_df[new_df["temperature"]==temp]
                X_train, y_train = df_temp[["PcEcPc", "liEcPc"]], df_temp[["conductivity"]]
                y_test_trial, y_trial_top, y_trial_bottom = None, None, None

            else: 
                TRAIN = False                    
                x_test_trial = pd.DataFrame(dict(PcEcPc = meshgrid_test_x, liEcPc = meshgrid_test_y))
                y_test_trial= trial_df["cond_mean"][trial_df["temperature"]==temp].values
                y_test_trial_min = trial_df["cond_min"][trial_df["temperature"]==temp].values
                y_test_trial_max = trial_df["cond_max"][trial_df["temperature"]==temp].values
                y_trial_top = y_test_trial_max - y_test_trial
                y_trial_bottom = y_test_trial - y_test_trial_min
                df_temp = old_df[old_df["temperature"]==temp]
                X_train, y_train = df_temp[["PcEcPc", "liEcPc"]], df_temp[["conductivity"]]

            # find the model of choice in ml_util file
            func = getattr(ml_util, generator)
            name = f"{generator}_{temp}"
            y_pred, best_est, model_train_score, result, width_average_score, width_average_trial = func(X_train, y_train, X_test, 
                                                                generator, temp, name, data_type=data_type, saving_model=True, num_iter=11,
                                                                saving_data = True, TRAIN=TRAIN, y_test_trial=y_test_trial, x_test_trial=x_test_trial, y_test_trial_top=y_trial_top,
                                                                y_test_trial_bottom = y_trial_bottom)

            best_params = pd.concat([best_params, pd.DataFrame([temp, model_train_score, best_est, width_average_score, width_average_trial]).T])

            all_results_for_plotting[generator][data_type][str(temp)]["train_model_score"] = model_train_score
            all_results_for_plotting[generator][data_type][str(temp)]["width_average_score"] = width_average_score
            all_results_for_plotting[generator][data_type][str(temp)]["width_average_trial"] = width_average_trial
                
            res = result.sort_values(by="conductivity", ascending=False)
            # getting the top percentile predictions
            best_result = pd.concat([best_result, res[:100]], axis=0, ignore_index=True)

        plot_highest_cond_plot(generator=generator, dataset=best_result, temperatures=temperatures,data_type=data_type,
                            file_name=f"top_res_{generator}", fsize=4, x_label = r"$\frac{PC}{EC+PC}$", y_label=r"$\frac{LiPF_{6}}{EC+ PC}$")

        # Save for every generator
        best_params.columns=["temperature", "model_score", "best_estimator", "width_average_score",  "width_average_trial"]

        save_dataset(best_params, generator=generator, data_type =data_type, separate=False, name=True)
        save_dataset(best_result, generator=generator, data_type =data_type, separate=False)


dumped = json.dumps(all_results_for_plotting, cls=NumpyEncoder)
with open("all_params_results.json", "w") as banana:
    json.dump(dumped, banana) 

