import sys
import os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from database import Database
from utils import *
from ml_util import *
import json

SAVE = False
mode = "single" 
generators = ["poly"] #linear
noise = 0.01  # Standard deviation of the noise
np.random.seed(0)

# Loading original dataset_type
data = Database()
df = data._initial_data
new_df = data._updated_data
## for comparison and performance evaluation 
trial_df = data._added_formulations
temperatures = data.sorted_temperatures
cols = ["PcEcPc", "liEcPc", "temperature"]

meshgrid_trial_x = [0.098316619, 0.098369507,0.098514549,0.098538967,0.098742297,0.098742364,0.098753423,0.098821919,0.098973291,0.09923281,0.10005633,0.10013993,0.100944341,0.109512134,0.110279441,0.110478765,0.323299518,0.327887589,0.330359239,0.336842455,0.33722314,0.344755564,0.346060962, 0.346792576]
meshgrid_trial_y = [0.87224212,0.301443329,0.252490738,0.786373263,0.274568453,0.85447359,0.797269059,0.282487301,0.290565767,0.865168137,0.797563723,0.823472427,0.83212906,0.292150801,0.271920445,0.262004695,0.280226177,0.300625252,0.273001711,0.272217056,0.301752066,0.29358397,0.284250375,0.287756431]


## order it based on percentile 
df_perc = df.copy()
df_perc = find_percentile(df_perc)
df_perc = df_perc.sort_values("percentile_rank")

data_types = ["old"]#, "new"]

all_results_for_plotting = {}
comparison_params = {"new": {"train_model_score": None}, "old": {"train_model_score": None}}


def best_poly_fit(df, new_df, cols, data_types, trial_df, meshgrid_trial_x=meshgrid_trial_x, meshgrid_trial_y=meshgrid_trial_y, 
                  df_perc=None, x_aug_test=None,temp=None, temperatures=None, degrees = np.arange(1, 10), 
                  min_rmse= 1e10 , min_deg =0, mode="all", generator="poly", 
                  regressor=BaggingRegressor(base_estimator=LinearRegression(), n_estimators=10, random_state=101), 
                  eval=False, n_repeat=50, num_cross_vals=4, PLOT=False):
    """find an optimal degree for polynomial regressor and fit the data with a regularized regressor 

    Args:
        df (dataframe): initial dataset
        new_df (dataframe): updated dataset
        cols (list): data features to be considered
        data_types (list): initial (old) or updated (new) dataset 
        trial_df (dataframe): data points for validation 
        meshgrid_trial_x (list, optional): formulations of choice for PC ratio. Defaults to meshgrid_trial_x.
        meshgrid_trial_y (list, optional): formulation of choise for LiPF6. Defaults to meshgrid_trial_y.
        df_perc (dataframe, optional): top percentile conductivity
        x_aug_test (dataframe, optional): exploited dataset.
        temp (integer, optional): selected temperature.
        temperatures (list, optional):list of temperatures.
        degrees (array, optional): range of degrees that can model pick. Defaults to np.arange(1, 10).
        min_rmse (float, optional): minimum root mean square error. Defaults to 1e10.
        min_deg (int, optional): minimum degree that model can take. Defaults to 0.
        mode (str, optional): modes of training, when single the model should be fit for a selected temperature otherwise all temperatures should be considered. Defaults to "all".
        generator (str, optional): polynomial or pairwise linear model. Defaults to "poly".
        regressor (model, optional): _description_. Defaults to BaggingRegressor(base_estimator=LinearRegression(), n_estimators=10, random_state=101).
        eval (bool, optional): if we need to add extra noise to evaluate the model performance. Defaults to False.
        n_repeat (int, optional): number of repetition. Defaults to 50.
        num_cross_vals (int, optional): number of cross validations . Defaults to 4.
    """

    # fit based on individual temperature
    if not mode =="all":
        if data_types=="old":
            df = df[df["temperature"] == temp]
            print(len(df))
        else:
            df = new_df[new_df["temperature"] == temp]

    X = df[cols]
    y = df[["conductivity"]]

    # Optimize and train based on initial dataset
    if data_types=="old":
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        
        if generator=="poly":
            rmses = []
            r2 = []
            degrees = np.arange(1, 10)
            min_rmse, min_deg = 1e10, 0

            # find an optimal degree of fit
            for deg in degrees:
                # Train features
                poly_features = PolynomialFeatures(degree=deg, include_bias=False)
                x_poly_train = poly_features.fit_transform(x_train)

                # Linear regression
                poly_reg = LinearRegression()
                poly_reg.fit(x_poly_train, y_train)

                # Compare with test data
                x_poly_test = poly_features.fit_transform(x_test)
                poly_predict = poly_reg.predict(x_poly_test)
                poly_mse = mean_squared_error(y_test, poly_predict)
                poly_rmse = np.sqrt(poly_mse)
                rmses.append(poly_rmse)
                r2.append(r2_score(y_test, poly_predict))

                # Cross-validation of degree
                if min_rmse > poly_rmse:
                    min_rmse = poly_rmse
                    min_deg = deg

            if mode == "all":
            # Plot and present results
                print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
                rsme_name = "all_rsme.png"
            else:
                print('Best degree {} with RMSE {} for temp {}'.format(min_deg, min_rmse, temp))
                rsme_name = f"min_degree_{temp}.png"
            if PLOT:
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
                ax.plot(degrees, rmses)
                ax.set_yscale('log')
                ax.set_xlabel('Degree')
                ax.set_ylabel('RMSE')
                plot_save_dir_degree = os.path.join(plot_dir, "augmented", "regression", "degree")
                plt.savefig(os.path.join(plot_save_dir_degree, rsme_name))
                plt.clf()

            # plot the outcome of fit for the predicted data
            y_test_sorted_index  = np.argsort(y_test.iloc[:,0]).tolist()
            if PLOT:
                plot_material(x=np.array([i for i in range(len(y_test))]), y=y_test.values[y_test_sorted_index], colour=None,x_label="Numbers", y_label="Conductivity", 
                            title=f"test versus prediction at {temp}", generator="poly",file_name=f"comp_{temp}.png", 
                            colorbar=False, plot=True, y_prime=poly_predict[y_test_sorted_index], xticks_range = [i for i in np.arange(0, 120, 10)])

            if isinstance(df_perc, pd.DataFrame):
                _ = cross_poly_calc(df_perc[cols], df_perc[["conductivity"]], min_deg, num_cross_vals, temp, generator, f"crossval_{generator}_{temp}.png")

            if eval:
                # performance evaluation of the model by adding some random noise to input features 
                x_test_sort = x_test
                y_test_sort = y_test
                x_train_all, y_train_all, x_test_all, y_test_all, y_pred_all= [], [], x_test_sort, np.zeros((len(x_test), n_repeat)), np.zeros((len(x_test), n_repeat))
                state_num = random.sample(range(200), n_repeat)
                best_poly_score = 0
                best_random_state = 0 
                for num, state in enumerate(state_num):
                    x_train_eval, _, y_train_eval, _ = train_test_split(X, y, test_size = 0.3, random_state=state)
                    x_train_all.append(x_train_eval)
                    y_train_all.append(x_train_eval)
                    y_test_all[:,num] = y_test_sort.values.flatten() + np.random.normal(0.0, 0.1,  len(x_test_sort))
                    _,_, y_pred_eval, _ = poly_fit(x=x_train_eval, y=y_train_eval, x_test=x_test_all, degree=min_deg)
                    y_pred_all[:,num] = y_pred_eval.flatten()

                y_error = np.zeros(len(x_test_all))
                for i in range(n_repeat):
                    for j in range(n_repeat):
                        y_error += (y_test_all[:,j] - y_pred_all[:,i])**2
                y_error /= n_repeat * n_repeat
                y_noise = np.var(y_test_all, axis=1)
                y_bias = (y_test_sort.values.flatten() - np.mean(y_pred_all, axis=1))**2
                y_var = np.var(y_pred_all, axis=1)
                print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
                        " + {3:.4f} (var) + {4:.4f} (noise)".format(
                        generator, np.mean(y_error), np.mean(y_bias), np.mean(y_var), np.mean(y_noise)
                    ))
                uncertainty_plot_poly([i for i in range(len(x_test_sort))], y_test_sort, x_train_all, y_train_all, y_pred_all, n_repeat, generator,
                                y_error, y_bias, y_noise, y_var, 1, temp)


        else: 
            _, y_pred_original = predefined_sklearn_method(x_train, y_train, x_test, regressor=regressor)
        
        # fit based on complete dataset at all temperatures
        if mode == "all":
            _,_, y_pred_original,_ = poly_fit(x_train, y_train, x_test, min_deg)
            for temper  in temperatures:    
                helper_calc(y_test[x_test["temperature"]==temper], y_pred_original[x_test["temperature"]==temper], x_test[x_test["temperature"]==temper], temp=f"all_{temper}", min_deg=min_deg, generator=generator)

        else:
            best_poly_score = 0
            best_random_state = 0 
            for num, state in enumerate(state_num):
                x_train_trial, x_test_trial , y_train_trial, y_test_trial = train_test_split(X, y, test_size = 0.3, random_state=state)
                _,poly_score_trial, y_pred_trial, _ = poly_fit(x_train_trial, y_train_trial, x_test_trial, min_deg)
                r2_poly = r2_score(y_test_trial, y_pred_trial)
                if (r2_poly > best_poly_score):
                    best_poly_score = r2_poly
                    best_random_state = state
                    print(f"the best random state is {state} with the r2 score of {best_poly_score} at temperature {temp} with the model score {poly_score_trial}.")

            x_train_best, x_test_best, y_train_best, y_test_best = train_test_split(X, y, test_size = 0.3, random_state=best_random_state)
            _,_, y_pred_best, _ = poly_fit(x_train_best, y_train_best, x_test_best, min_deg)
            helper_calc(y_test_best, y_pred_best, x_test_best, temp, min_deg, generator, PLOT=PLOT)

    # main calculation
    if not mode=="all":
        if generator=="poly":
            if data_types=="old":
                    temp_col = [temp for i in range(len(meshgrid_trial_x))]
                    x_test_trial = pd.DataFrame(dict(PcEcPc = meshgrid_trial_x, liEcPc = meshgrid_trial_y,temperature= temp_col))
                    y_test_trial= trial_df["cond_mean"][trial_df["temperature"]==temp].values
                    y_test_trial_min = trial_df["cond_min"][trial_df["temperature"]==temp].values
                    y_test_trial_max = trial_df["cond_max"][trial_df["temperature"]==temp].values
                    y_trial_top = y_test_trial_max - y_test_trial
                    y_trial_bottom = y_test_trial - y_test_trial_min

                    model, score, y_pred , _= poly_fit(X, y, x_test=x_aug_test,
                                             x_test_trial=x_test_trial, y_test_trial=y_test_trial, 
                                             y_test_trial_top=y_trial_top, y_test_trial_bottom=y_trial_bottom,
                                             degree=min_deg, temp=temp, data_types=data_types, save_result=True, TRIAL=True)

            else:
                model, score, y_pred , _= poly_fit(X, y, x_test=x_aug_test,
                                             x_test_trial=None, y_test_trial=None, 
                                             y_test_trial_top=None, y_test_trial_bottom=None,
                                             degree=min_deg, temp=temp, data_types=data_types, save_result=True, TRIAL=False)
            print(f"R_squared score is {score}")

        else:
            score, y_pred = predefined_sklearn_method(X, y, x_aug_test, regressor=regressor)
        return model, min_deg, score, y_pred

    else: 
        return min_deg


def poly_fit(x, y, x_test, x_test_trial=None, y_test_trial=None, y_test_trial_top=None, y_test_trial_bottom=None, degree=5, 
             temp=None, data_types="old", save_result = False, TRIAL=False):
    """_summary_

    Args:
        x (array): input features
        y (array): output feature
        x_test (array): input of test dataset
        x_test_trial (array, optional): optional input testset for comparison purposes. Defaults to None.
        y_test_trial (array, optional): optional output testset for comparison purposes. Defaults to None.
        y_test_trial_top (array, optional): optional upper error for output testset for comparison purposes. Defaults to None.
        y_test_trial_bottom (array, optional): optional lower error for output testset for comparison purposes. Defaults to None.
        degree (int, optional): an optimal polynomial degree. Defaults to 5.
        temp (_type_, optional): specific temperature or any temperature can be selected. Defaults to None.
        data_types (str, optional): type of dataset (before and after one shot). Defaults to "old".
        save_result (bool, optional): is result should be saved. Defaults to False.
        TRIAL (bool, optional): if any comparison should be applied. Defaults to False.

    Returns:
        _type_: the trained model, the r_squared, predictions and the chosen degree
    """

    if data_types=="old":
        # this script is used for training with initial dataset
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-4))
        
    else:
        # evaluating the model applied on the updated dataset by retreieving the same setting
        model_retrieve = joblib.load(open(os.path.join("data", "augmented", "regression", "separate_datasets", "poly", "old", f"poly_model_{temp}.pkl"), "rb"))
        degree = model_retrieve.get_params()["polynomialfeatures__degree"] 
        alpha = model_retrieve.get_params()["ridge__alpha"]
        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))

    model.fit(x, y)
    model_score = model.score(x,y)
    y_pred = model.predict(x_test)

    if TRIAL:
        y_pred_trial = model.predict(x_test_trial)
        do_comparison_confidence_interval("poly", temp, f"poly_{temp}", y_test_trial=y_test_trial, 
                                        y_test_trial_top=y_test_trial_top, y_test_trial_bottom=y_test_trial_bottom, 
                                        y_pred_trial=y_pred_trial, data_type=data_types)

    if save_result:
        save_model(model, generator="poly", data_type=data_types, separate=True, temperature=temp, dumping=True)

    return model, model_score, y_pred, degree


    
def predefined_sklearn_method(x, y, x_test, regressor=BaggingRegressor(base_estimator = LinearRegression(), n_estimators=10, random_state=101)):
    """_summary_

    Args:
        x (array): input trainset
        y (array): output trainset
        x_test (array): input testset
        regressor (_type_, optional): _description_. Defaults to BaggingRegressor(base_estimator = LinearRegression(), n_estimators=10, random_state=101).

    Returns:
        _type_: model score, predictions
    """
    regressor = regressor
    regressor.fit(x, y)
    y_pred = regressor.predict(x_test)
    score = regressor.score(x,y)
    print(score)
    return score, y_pred



augment_all = pd.DataFrame()
top_temp_augment = pd.DataFrame()

# creating 10000 new formulations and exploiting the system
step_size= 100
pcecpc_iterator = ((max(df["PcEcPc"])+0.015) - min(df["PcEcPc"])) / step_size
liecpc_iterator = ((max(df["liEcPc"])+0.015) - min(df["liEcPc"])) / step_size
pcecpc_test = np.array([i for i in np.arange(min(df["PcEcPc"]), (max(df["PcEcPc"]+0.015)), pcecpc_iterator)])
liecpc_test = np.array([i for i in np.arange(min(df["liEcPc"]), (max(df["liEcPc"]+0.015)), liecpc_iterator)])
meshgrid_test_x, meshgrid_test_y = np.meshgrid(pcecpc_test, liecpc_test)


for data_type in data_types:
    all_results_for_plotting[data_type] = {}
    best_params = pd.DataFrame()
    for generator in generators:
        all_results_for_plotting[data_type][generator] = {}
        print(generator + data_type) 
        if mode=="all":
            min_degree = best_poly_fit(df, new_df=new_df,cols=cols, data_types=data_type, x_aug_test=None,temp=None, temperatures=temperatures, degrees = np.arange(1, 10), min_rmse= 1e10 , min_deg =0, mode=mode, generator=generator)
        for temperature in temperatures:
            
            all_results_for_plotting[data_type][generator][str(temperature)] = {}
            augmented_data, x_test_grid = pd.DataFrame(), pd.DataFrame()
            grid_temperature = np.array([temperature for i in range(step_size*step_size)])
            x_test_grid = pd.DataFrame(dict(PcEcPc = meshgrid_test_x.ravel(), liEcPc = meshgrid_test_y.ravel(), temperature = grid_temperature))
            
            if mode=="all":
                if generator=="poly":
                    _, y_test = poly_fit(x = df[cols], y=df[["conductivity"]], x_test=x_test_grid, degree=min_degree)
                else:
                    _, y_test = predefined_sklearn_method(x = df[cols], y=df[["conductivity"]], x_test=x_test_grid)
            else:
                model, min_deg, score, y_test =  best_poly_fit(df,new_df=new_df, cols=cols, 
                                                    data_types=data_type,trial_df=trial_df,
                                                  meshgrid_trial_x=meshgrid_trial_x, meshgrid_trial_y=meshgrid_trial_y,
                                                  df_perc=df_perc, x_aug_test= x_test_grid, temp=temperature, mode = mode, generator=generator, eval=True)
                best_params = pd.concat([best_params, pd.DataFrame([temperature, score, model]).T])

            
            augmented_data = augmented_data.append(pd.concat([x_test_grid, pd.DataFrame(y_test)],axis=1))
            augmented_data.columns = [*augmented_data.columns[:-1], 'conductivity']
            save_dataset(dataset=augmented_data, augmentation_method="regression", generator=f"{generator}", separate=True, temperature=temperature, data_type=data_type)

            augment_all = augment_all.append(pd.concat([x_test_grid, pd.DataFrame(y_test)], axis=1))
            augment_all = augment_all.rename(columns={"0":"conductivity"})
            save_dataset(dataset=augment_all, augmentation_method="regression", generator=f"{generator}", separate=False, temperature=temperature, data_type=data_type)

            top_temp_augment = top_temp_augment.append(augmented_data.sort_values(by="conductivity", ascending=False, ignore_index=True)[:100])
            top_temp_augment = top_temp_augment.rename(columns={"0":"conductivity"})
            all_results_for_plotting[data_type][generator][str(temperature)]["train_model_score"]= score
            
        best_params.columns=["temperature", "model_score", "best_estimator"]
        save_dataset(best_params, generator=generator, data_type =data_type, separate=False, name=True)

        # Visualising the predicitons
        plot_material(x=top_temp_augment["PcEcPc"], y=top_temp_augment["liEcPc"], colour=top_temp_augment["conductivity"],  title=f"top_exp_with_{generator}",
                        file_name=f"{generator}_{data_type}_top_experiment_base_temperature.png", dataset_type="augmented",
                        annotate_point=True, ann=top_temp_augment["temperature"], point_size=100, generator=generator)

        plot_highest_cond_plot(generator=generator, dataset=top_temp_augment, temperatures=temperatures,data_type=data_type,
                                file_name=f"top_res_{generator}", fsize=4, x_label = r"$\frac{PC}{EC+PC}$", y_label=r"$\frac{LiPF_{6}}{Ec + PC}$")

if SAVE:
    dumped = json.dumps(all_results_for_plotting, cls=NumpyEncoder)
    with open("all_params_results_poly_initial.json", "w") as banana:
        json.dump(dumped, banana)






