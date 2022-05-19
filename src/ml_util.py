from pyexpat import model
from turtle import color
from matplotlib.pyplot import title
from numpy import size
from utils import *
from mapie.regression import MapieRegressor
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import KFold
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor
from typing import Union
from mapie.subsample import Subsample
from skopt import BayesSearchCV
from skopt.plots import plot_objective
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

poly_equations = dict()
np.random.seed(0)

def save_generator_plot(generator, analysis, name, data_type):
    """creating save path for generated plots

    Args:
        generator (str): name of estimator
        analysis (analysis): type of analysis
        name (str): name of the plot
        data_type (str): type of data
    """
    plot_save_dir = os.path.join(plot_dir, "augmented", "regression", generator, analysis, data_type)
    check_folder = os.path.isdir(plot_save_dir)
    if not check_folder:
        os.makedirs(plot_save_dir)
    plt.savefig(os.path.join(plot_save_dir, f"{generator}_{name}.png"))
    plt.savefig(os.path.join(plot_save_dir, f"{generator}_{name}.svg"))
    plt.clf()
    
    
def figure_identity(title, x_label, y_label, square = False, x_ticks = None, figsize=8, y_ticks=None, ROTATION=False):
    """Initiallizing general plotting format

    Args:
        title (str): title of plot
        x_label (str): label of x axis
        y_label (str): label of y axis
        square (bool, optional): if the plot should be squared. Defaults to False.
        x_ticks (list, optional): list of x axis ticks. Defaults to None.
        figsize (int, optional): size of figure. Defaults to 8.
        y_ticks (list, optional): list of y axis ticks. Defaults to None.
        ROTATION (bool, optional): if the axis ticks should be rotated. Defaults to False.
    """
    plt.style.use(['nature', 'science', 'no-latex'])
    plt.figure(figsize=(figsize,figsize))
    if title:
        plt.title(title, fontsize=15, pad=12)
    if square:
        plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    if x_ticks: 
        plt.xticks(x_ticks, size=9, rotation=90)
    else:
        if ROTATION: 
            plt.xticks(size=9, rotation=90)
        else: 
            plt.xticks(size=9)
    if (y_ticks is not None) and len(y_ticks)>0:
        plt.yticks(y_ticks, size=9)
    else:
        plt.yticks(size=9)
    plt.xlabel(x_label,fontdict=dict(size=14.5))
    plt.ylabel(y_label,fontdict=dict(size=14.5)) 
    

def uncertainty_plot(y_pred, y_pis):
    """drawing prediction interval

    Args:
        y_pred (dataframe): the predictions
        y_pis (dataframe): upper and lower bound of predictions
    """
    y_pred_filled = np.sort(y_pred)
    y_pred_fill_index = np.argsort(y_pred).tolist()
    y_1_fill = np.array(y_pis[:,1,0]).flatten()[y_pred_fill_index]
    y_2_fill = np.array(y_pis[:,0,0]).flatten()[y_pred_fill_index]
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.scatter([i for i in range(len(y_pred))], y_pred_filled.flatten(), s=4, color="green")
    plt.fill_between([i for i in range(len(y_pred))], y_1_fill, y_2_fill, alpha=0.4, color="green")
    plt.legend()


def prediction_confidence_plot(y_pred, y_pis, y_test):
    """prepration of  prediction interval plot while including the test data

    Args:
        y_pred (dataframe): predicted dataset
        y_pis (dataframe): upper and lower dataset of predictions
        y_test (dataframe): test dataset
    """
    y_pred_filled = np.sort(y_pred)
    y_pred_fill_index = np.argsort(y_pred).tolist()
    y_1_fill = np.array(y_pis[:,1,0]).flatten()[y_pred_fill_index]
    y_2_fill = np.array(y_pis[:,0,0]).flatten()[y_pred_fill_index]
    y_test_filled = np.array(y_test).flatten()[y_pred_fill_index]
    plt.plot([i for i in range(len(y_pred))], y_pred_filled.flatten(), label="Prediction intervals", color="green")
    plt.plot([i for i in range(len(y_pred))], y_test_filled, label="True confidence intervals", color="orange")
    plt.fill_between([i for i in range(len(y_pred))], y_1_fill, y_2_fill, alpha=0.4, color="green")
    plt.legend()


def do_prediction_confidence_interval(generator, temp, name, y_pred, y_pis, y_test, data_type):
    """uncertainty using predicition interval 

    Args:
        generator (str): type of regressor
        temp (float): temperature of choice
        name (str): plot name
        y_pred (dataframe): predicted dataset
        y_pis (dataframe): upper and lower dataset of predictions
        y_test (dataframe): test dataset
        data_type (str): type of data
    """

    title = f"Jacknife+ for temperature {temp} with {generator}"
    figure_identity(title=title, x_label="# of experiment", y_label=r"$\sigma$ [$\frac{1}{\Omega cm}$]", figsize=4, ROTATION=True)
    prediction_confidence_plot(y_pred=y_pred, y_pis=y_pis, y_test=y_test)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-2,2))
    save_generator_plot(generator, analysis="intervals", name=name, data_type=data_type)

def draw_boxplot(generator_position, data, offset,edge_color):
    """box plot creation

    Args:
        generator_position (float): position of each regressors
        data (dataframe): dataset
        offset (float): offset of the bars
        edge_color (str): color of choice
    """
    pos = generator_position + offset 
    bp = plt.boxplot(data.values(), positions= pos, showmeans=False, widths=0.2)

    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)


def do_draw_boxplot(generator, name, x_label, y_label, generator_position, data, offset, edge_color, fill_color, data_type, title):
    """draw box plot with right format 

    Args:
        generator (str): type of the regressor
        name (str): plot name 
        x_label (str): x axis label
        y_label (str): y axis label
        generator_position (float): position of each regressors
        data (dataframe): dataset
        offset (float): offset of the bars
        edge_color (str): color of choice for the bar edges 
        fill_color (str): color of choice for filling the bars
        data_type (str): dataset type
        title (str): plot title
    """
    figure_identity(title=title, x_label=x_label, y_label=y_label)
    draw_boxplot(generator_position, data, offset, edge_color, fill_color)
    save_generator_plot(generator, analysis="comparion", name=name, data_type=data_type)
   

def comparison_error_plot(y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial):
    """tempelate for comparison plot incorporating actual measurements with their errors and the predictions 

    Args:
        y_test_trial (array):the mean of conductivity for actual measurements
        y_test_trial_top (array): The highest bound of the measured conductivity
        y_test_trial_bottom (array): The lowest bound of the measured conductivity
        y_pred_trial (array): The predicted conductivity
    """
    if len(y_pred_trial.shape) >1:
        y_pred_trial = y_pred_trial[:,0]
    y_pred_sort_index = np.argsort(y_pred_trial).tolist()
    y_pred_trial_sort = np.array(y_pred_trial).flatten()[y_pred_sort_index]
    y_test_trial_sort = np.array(y_test_trial).flatten()[y_pred_sort_index]
    y_test_trial_top_sort = np.array(y_test_trial_top).flatten()[y_pred_sort_index]
    y_test_trial_bottom_sort = np.array(y_test_trial_bottom).flatten()[y_pred_sort_index]
    
    plt.errorbar(range(len(y_test_trial_sort)), y_test_trial_sort, yerr=(y_test_trial_bottom_sort, y_test_trial_top_sort), fmt="o",
                 color="#ec8100", ecolor="#ffaf4f", elinewidth=1.5, capsize=1, label="Experimetally Measured")
    plt.scatter(range(len(y_test_trial_sort)), y_pred_trial_sort, marker="x", color="royalblue", label="Predicted", s=10, zorder=3)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-2,2))
    plt.legend()


def comparison_error_scatter_plot(y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial):
    """tempelate for scatter comparison plot incorporating actual measurements with their errors 

    Args:
        y_test_trial (array): the mean of conductivity for actual measurements
        y_test_trial_top (array): The highest bound of the measured conductivity
        y_test_trial_bottom (array): The lowest bound of the measured conductivity
        y_pred_trial (array): The predicted conductivity
    """
    y_pred_sort_index = np.argsort(y_pred_trial).tolist()
    y_pred_trial_sort = np.array(y_pred_trial).flatten()[y_pred_sort_index]
    y_test_trial_sort = np.array(y_test_trial).flatten()[y_pred_sort_index]
    y_test_trial_top_sort = np.array(y_test_trial_top).flatten()[y_pred_sort_index]
    y_test_trial_bottom_sort = np.array(y_test_trial_bottom).flatten()[y_pred_sort_index]

    plt.errorbar(y_pred_trial_sort, y_test_trial_sort, yerr=(y_test_trial_bottom_sort, y_test_trial_top_sort), fmt="o",
                 color="#ec8100", ecolor="#ffaf4f", elinewidth=1, capsize=0.5)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(-2,2))
    plt.legend()


def do_comparison_error_scatter_plot(generator, name, y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial, data_type):
    """assemble the comparison error plot 

    Args:
        generator (str): the regressor of choice 
        name (str): name of the plt
        y_test_trial (array): the mean of conductivity for actual measurements
        y_test_trial_top (array): The highest bound of the measured conductivity
        y_test_trial_bottom (array): The lowest bound of the measured conductivity
        y_pred_trial (array): The predicted conductivity
        data_type (str): type of the data
    """
    bottom_tick_y = round(np.min(y_test_trial_bottom)-0.0005, 5)
    top_tick_y = round(np.max(y_test_trial_top) + 0.0005, 5)
    freq_y = (top_tick_y - bottom_tick_y)/6
    y_ticks = [round(bottom_tick_y+ freq_y*(i+1), 5) for i in range(6)]
    figure_identity(title=None, x_label="Predicted", y_label=r"Measured $\sigma$ [$\frac{1}{\Omega cm}$]", figsize=3, y_ticks=y_ticks, x_ticks=y_ticks)
    comparison_error_scatter_plot(y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial)
    save_generator_plot(generator, analysis="comparison_scatter", name=name, data_type=data_type)


def do_comparison_confidence_interval(generator, temp, name, y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial, data_type):
    """plotthing function for confidence interval

    Args:
        generator (str): the regressor of choice 
        temp (float): temperature of choice
        name (str): plot name 
        y_test_trial (array): the mean of conductivity for actual measurements
        y_test_trial_top (array): The highest bound of the measured conductivity
        y_test_trial_bottom (array): The lowest bound of the measured conductivity
        y_pred_trial (array): The predicted conductivity
        data_type (str): type of the data
    """
    if (np.min(y_pred_trial)-0.0005) < 0:
        bottom_tick = 0.00000
    else:
        bottom_tick = round(np.min(y_pred_trial)-0.0005, 5)
    top_tick = round(np.max(y_pred_trial) + 0.0005, 5)
    freq = (top_tick - bottom_tick)/4
    y_ticks = [round(bottom_tick+ freq*(i+1), 6) for i in range(4)]

    figure_identity(title=None, x_label="# of experiment", y_label=r"$\sigma$ [$\frac{1}{\Omega cm}$]", figsize=3, y_ticks=y_ticks)
    comparison_error_plot(y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial)
    save_generator_plot(generator, analysis="comparison", name=name, data_type=data_type)


def plot_contour(x_train, result, generator, temp, data_type, UNCERTAINTY=True):
    """contour plot function for showing individual predictions/prediction intervals from a trained model

    Args:
        x_train (array): train features
        result (array): test features
        generator (str): the regressor of choice 
        temp (float): temperature of choice
        data_type (str): type of the data
        UNCERTAINTY (bool, optional): whether we are plotting the prediction intervals or the predictions. Defaults to True.
    """
    x_real, y_real = x_train["PcEcPc"], x_train["liEcPc"]
    plot_material(x=result["PcEcPc"], y=result["liEcPc"], colour=result["conductivity"]*1000,x_label = r"$\frac{PC}{EC+PC}$" ,
                y_label=r"$\frac{LiPF_{6}}{EC + PC}$", file_name=f"{generator}_{temp}_scaled", dataset_type="augmented", 
                generator=generator, figsize=(4,4), contour=True, square=True, colorbar=True, real_data=True, x_real=x_real, y_real=y_real, 
                retrain=True, contour_level=20, data_type=data_type, uncertaiinty=False, ROTATION=True)
    if UNCERTAINTY:
        # plot contour for uncertainty measurement
        plot_material(x=result["PcEcPc"], y=result["liEcPc"], colour=result["PI_uncertainty"]*1000,x_label = r"$\frac{PC}{EC+PC}$", y_label=r"$\frac{LiPF_{6}}{Ec + PC}$",
                    file_name=f"{generator}_{temp}", dataset_type="augmented", generator=generator,figsize=(4,4),
                    contour=True, square=True, colorbar=True, real_data=True, x_real=x_real, y_real=y_real, 
                    contourname=True, contour_level=20, data_type=data_type, uncertaiinty=True, ROTATION=True)


def do_uncertenty_plot(generator, temp, name, y_pred, y_pis, data_type):
    """assemble prediction interval plot for preducted versus actual measurements

    Args:
        generator (str): the regressor of choice 
        temp (float): temperature of choice
        name (str): plot name 
        y_pred (array): predicted conductivity
        y_pis (array):n an array containty lower and higher interval of predictions
        data_type (str): type of the data
    """
    title = f"Jackknife+ for temperature {temp}"
    figure_identity(title=title, x_label="# of experiment", y_label=r"Predicted $\sigma$ [$\frac{1}{\Omega cm}$]", figsize=4, ROTATION=True)
    uncertainty_plot(y_pred=y_pred, y_pis=y_pis)
    save_generator_plot(generator, analysis="uncertainty", name=name, data_type=data_type)


def do_objective_plot(model, dimensions, generator, temp, name, data_type):
    """objective plot function for showing the trend in hyperparameter tuning

    Args:
        model (class): the regressor of choice 
        dimensions (list): list of all the parameters to be tuned
        generator (str): the regressor of choice 
        temp (float): temperature of choice
        name (str): plot name 
        data_type (str): type of the data
    """
    title = f"Hypertuned params at {temp}°"
    figure_identity(title=title, x_label="", y_label="")
    plot_objective(model.optimizer_results_[0], dimensions=dimensions, n_minimum_search=int(1e8))
    save_generator_plot(generator, analysis="hypertune", name=name, data_type= data_type)


def prepare_saving(temp, x_test, y_pred, y_pis):
    result = pd.DataFrame()
    result = pd.concat([x_test, pd.Series([temp for _ in range(len(y_pred))]), pd.Series(y_pred),
                        pd.Series(y_pis[:,1,0]-y_pis[:,0,0]), pd.Series(y_pis[:,1,0]), pd.Series(y_pis[:,0,0])
                        ], axis=1, ignore_index=True)
    result.columns=["PcEcPc", "liEcPc", "temperature", "conductivity", "PI_uncertainty", "upper_confidance", "lower_confidance"]
    return result

# Prediction intervals must account for both the uncertainty in estimating the population mean, plus the random variation of the individual values. 
# So a prediction interval is always wider than a confidence interval.
#Average t*StDev*(sqrt(1+(1/n))), where t is a tabled value from the t distribution which depends on the confidence level and sample size.
#https://www.graphpad.com/support/faq/the-distinction-between-confidence-intervals-prediction-intervals-and-tolerance-intervals/
def mapie_training(best_est, X_train, y_train, generator, saving_model, temp, x_test, name, grad_bayes, grad_params, 
                   saving_data, data_type, HYPTERPARAM=True, UNCERTAINTY=True, y_test_trial = None, x_test_trial=None,
                   y_test_trial_top=None, y_test_trial_bottom=None, trial=False):
    """prediction with the model agnostic prediction interval incorporated with the machine learning of the choice.

    Args:
        best_est (class): hypertuned surrogate model
        X_train (array): train dataset
        y_train (array): label of the train dataset
        generator (str): the regressor of choice 
        saving_model (bool): if the model should be saved
        temp (float): temperature of choice
        x_test (array): test dataset
        name (str): plot name 
        grad_bayes (class): the model of interest
        grad_params (list): parameters used for hyperparameter tuning 
        saving_data (bool): if dataset should be saved
        data_type (str): type of the data
        HYPTERPARAM (bool, optional): if objective function for the tuned model need to be plotted. Defaults to True.
        UNCERTAINTY (bool, optional): if prediction interval of the calibrated model need to be plotted. Defaults to True.
        y_test_trial (array): the mean of conductivity for actual measurements. Defaults to None.
        x_test_trial (array, optional): the exact formualated compositions for actual measurements. Defaults to None
        y_test_trial_top (array): The highest bound of the measured conductivity. Defaults to None.
        y_test_trial_bottom (array): The lowest bound of the measured conductivity. Defaults to None.
        trial (bool, optional): if evaluation should be done. Defaults to False.
    """

    model= MapieRegressor(best_est, method = "plus", cv=-1, agg_function="median", n_jobs=-1)
    model.fit(X_train, y_train.values.reshape(len(y_train),))
    if saving_model:
        save_model(model, generator=generator, data_type= data_type, separate=True, temperature=temp, dumping=True)
    y_pred, y_pis = model.predict(x_test, alpha=0.05)
    model_score = model.score(X_train, y_train)
    #### adjusted r2 score for removing the biased estimation of r2
    p = 3 # independent values 
    Adj_r2 = 1-(1-model_score)*(len(X_train)-1)/(len(X_train)-p-1)
    width_average_score = (y_pis[:,1,0]-y_pis[:,0,0]).mean()
    print(
        f"score comes from {generator} after optimization in degree {model_score}")
    if trial:
        y_pred_trial, y_pis_trial = model.predict(x_test_trial, alpha=0.05)
        width_average_trial = (y_pis_trial[:,1:0]-y_pis_trial[:,0:0]).mean()
        do_comparison_confidence_interval(generator, temp, name, y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial, data_type)
        do_comparison_error_scatter_plot(generator, temp, name, y_test_trial, y_test_trial_top, y_test_trial_bottom, y_pred_trial, data_type)
        do_prediction_confidence_interval(generator, temp, name, y_pred_trial, y_pis_trial, y_test_trial, data_type)
    else:
        width_average_trial = None

    if HYPTERPARAM:
        do_objective_plot(grad_bayes, list(grad_params.keys()), generator, temp, name, data_type)

    do_uncertenty_plot(generator, temp, name, y_pred, y_pis, data_type)
    result = prepare_saving(temp, x_test, y_pred, y_pis)
    plot_contour(x_train=X_train, result=result, generator=generator, temp=temp, UNCERTAINTY=UNCERTAINTY, data_type=data_type)
    
    if saving_data:
      save_dataset(result, generator=generator, separate=True, temperature=temp, data_type=data_type)

    return y_pred, y_pis, model_score, result, width_average_score, width_average_trial


def poly_fit(X_train, y_train, x_test, generator, temp, name, data_type,
             n_splits=3, saving_model=False, saving_data=False,num_iter=20, TRAIN=False, y_test_trial = None, 
             x_test_trial=None, y_test_trial_top=None, y_test_trial_bottom=None, EQ=True):
    """regularized polynomial regressor

    Args:
        X_train (array): train dataset
        y_train (array): label of the train dataset
        x_test (array): test dataset
        generator (str): the regressor of choice 
        temp (float): temperature of choice
        name (str): plot name 
        data_type (str): type of the data
        n_splits (int, optional): number of split for cross validation. Defaults to 3.
        saving_model (bool, optional): if the trained model should be saved. Defaults to False.
        saving_data (bool, optional): if the dataset should be saved. Defaults to False.
        num_iter (int, optional): number of iteration for bayes hyperparameter tuning. Defaults to 20.
        TRAIN (bool, optional): if the trainset should be fit or the pretrained model should be used. Defaults to False.
        y_test_trial (array): the mean of conductivity for actual measurements. Defaults to None.
        x_test_trial (array, optional): the exact formualated compositions for actual measurements. Defaults to None
        y_test_trial_top (array): The highest bound of the measured conductivity. Defaults to None.
        y_test_trial_bottom (array): The lowest bound of the measured conductivity. Defaults to None.
        EQ (bool, optional): if the trained equations should be stored. Defaults to True.

    """

    cv = KFold(n_splits=n_splits, shuffle=False)

    if TRAIN:
        model = make_pipeline(PolynomialFeatures(), Ridge(random_state=101))
        params = {"polynomialfeatures__degree" : [i for i in range(2, 7)], "ridge__alpha": [0.0005, 0.001, 0.005, 0.007, 0.01, 0.03]}
        model_bayes = BayesSearchCV(model, params, n_iter=num_iter, cv=cv, random_state=101, n_jobs=-1,
                                    scoring="r2", return_train_score=True)
        model_bayes.fit(X_train, y_train.values.reshape(len(y_train),))
        best_est = model_bayes.best_estimator_
        model_coeeficients = best_est.steps[1][1].coef_
        model_intercept = best_est.steps[1][1].intercept_
        model_names = best_est.steps[0][1].get_feature_names()

        if EQ: 
            eq = f"y = {round(model_intercept, 4)} + {round(model_coeeficients[0], 4)}*{model_names[0]} + {round(model_coeeficients[1], 4)}*{model_names[1]} + {round(model_coeeficients[2], 4)}*{model_names[2]} + {round(model_coeeficients[3], 4)}*{model_names[3]} + {round(model_coeeficients[4], 4)}*{ model_names[4]} + {round(model_coeeficients[5], 4)}*{model_names[5]} + {round(model_coeeficients[6], 4)}*{ model_names[6]} + {round(model_coeeficients[7], 4)}*{ model_names[7]} + {round(model_coeeficients[8], 4)}*{ model_names[8]} + {round(model_coeeficients[9], 4)}*{ model_names[9]}"
            poly_equations.update({f"poly_fit_{temp}": eq})

            with open(os.path.join("data", "augmented", "regression", "combined_datasets", "poly_fit", "new", "poly_fit_equation.json"), "w") as f: 
                json.dump(poly_equations, f)

        HYPTERPARAM, trial = True, False

    else:
        model_bayes = joblib.load(open(os.path.join("data", "augmented", "regression", "separate_datasets", "poly_fit", "new", f"poly_fit_model_{temp}.pkl"), "rb"))
        degree = model_bayes.get_params()["estimator__polynomialfeatures__degree"]
        alpha = model_bayes.get_params()["estimator__ridge__alpha"]
        best_est = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha, random_state=101))
        params = {"degree": degree, "alpha":alpha}
        HYPTERPARAM=False
        trial = True

    y_pred, _, model_train_score, result, width_average_score, width_average_trial = mapie_training(best_est,
                                                        X_train, y_train, generator, saving_model, 
                                                        temp, x_test, name, model_bayes, params, saving_data, data_type=data_type, 
                                                        HYPTERPARAM=HYPTERPARAM, y_test_trial = y_test_trial, x_test_trial=x_test_trial,
                                                        y_test_trial_top=y_test_trial_top, y_test_trial_bottom=y_test_trial_bottom, trial=trial)

    return y_pred, best_est, model_train_score, result, width_average_score, width_average_trial



def extratrees_fit(X_train, y_train, x_test, generator, temp, name, data_type,
             n_splits=3, saving_model=False, saving_data=False, num_iter=15, TRAIN=False, y_test_trial = None, 
             x_test_trial=None, y_test_trial_top=None, y_test_trial_bottom=None): 
    """extratree regressor

    Args:
        X_train (array): train dataset
        y_train (array): label of the train dataset
        x_test (array): test dataset
        generator (str): the regressor of choice 
        temp (float): temperature of choice
        name (str): plot name 
        data_type (str): type of the data
        n_splits (int, optional): number of split for cross validation. Defaults to 3.
        saving_model (bool, optional): if the trained model should be saved. Defaults to False.
        saving_data (bool, optional): if the dataset should be saved. Defaults to False.
        num_iter (int, optional):  number of iteration for bayes hyperparameter tuning. Defaults to 15.
        TRAIN (bool, optional): if the trainset should be fit or the pretrained model should be used. Defaults to False.
        y_test_trial (array): the mean of conductivity for actual measurements. Defaults to None.
        x_test_trial (array, optional): the exact formualated compositions for actual measurements. Defaults to None
        y_test_trial_top (array): The highest bound of the measured conductivity. Defaults to None.
        y_test_trial_bottom (array): The lowest bound of the measured conductivity. Defaults to None.

    """

    cv = KFold(n_splits=n_splits, shuffle=False)
    if TRAIN:
        trees_model = ExtraTreesRegressor(random_state=101)
        trees_params = {"n_estimators": [i for i in range(5, 12)], "max_depth": [i for i in range(2, 9)]}
        tree_bayes = BayesSearchCV(trees_model, trees_params, n_iter=num_iter, cv=cv, random_state=101, n_jobs=-1, 
                                scoring="r2", return_train_score = True)
        
        tree_bayes.fit(X_train, y_train.values.reshape(len(y_train),))
        best_est = tree_bayes.best_estimator_
        HYPTERPARAM=True
        trial = False
    else: 

        tree_bayes = joblib.load(open(os.path.join("data", "augmented", "regression", "separate_datasets", "extratrees_fit", "new", f"extratrees_fit_model_{temp}.pkl"), "rb"))
        max_depth = tree_bayes.get_params()["estimator"].max_depth
        n_estimators = tree_bayes.get_params()["estimator"].n_estimators
        best_est = ExtraTreesRegressor(random_state=101, verbose=0, max_depth=max_depth, n_estimators=n_estimators)
        trees_params = {"max_depth": max_depth, "n_estimators":n_estimators}
        HYPTERPARAM=False
        trial = True

    y_pred, _, model_train_score, result, width_average_score, width_average_trial = mapie_training(best_est,
                                                        X_train, y_train, generator, saving_model, 
                                                        temp, x_test, name, tree_bayes, trees_params, saving_data, data_type=data_type, 
                                                        HYPTERPARAM=HYPTERPARAM, y_test_trial = y_test_trial, x_test_trial=x_test_trial,
                                                        y_test_trial_top=y_test_trial_top, y_test_trial_bottom=y_test_trial_bottom, trial=trial)
    
    return y_pred, best_est, model_train_score, result, width_average_score, width_average_trial




def bagging_extra_fit(X_train, y_train, x_test, generator, temp, name, data_type,
             n_splits=3, saving_model=False, saving_data=False, num_iter=15, TRAIN=False, y_test_trial = None, 
             x_test_trial=None, y_test_trial_top=None, y_test_trial_bottom=None): 
    """bagging estimator with extratree regressor

    Args:
        X_train (array): train dataset
        y_train (array): label of the train dataset
        x_test (array): test dataset
        generator (str): the regressor of choice 
        temp (float): temperature of choice
        name (str): plot name 
        data_type (str): type of the data
        n_splits (int, optional): number of split for cross validation. Defaults to 3.
        saving_model (bool, optional): if the trained model should be saved. Defaults to False.
        saving_data (bool, optional): if the dataset should be saved. Defaults to False.
        num_iter (int, optional):  number of iteration for bayes hyperparameter tuning. Defaults to 15.
        TRAIN (bool, optional): if the trainset should be fit or the pretrained model should be used. Defaults to False.
        y_test_trial (array): the mean of conductivity for actual measurements. Defaults to None.
        x_test_trial (array, optional): the exact formualated compositions for actual measurements. Defaults to None
        y_test_trial_top (array): The highest bound of the measured conductivity. Defaults to None.
        y_test_trial_bottom (array): The lowest bound of the measured conductivity. Defaults to None.

    """

    cv = KFold(n_splits=n_splits, shuffle=False)
    if TRAIN:
        trees_model = BaggingRegressor(ExtraTreeRegressor(),random_state=101)
        trees_params =  {"n_estimators": [i for i in range(2, 11)], "max_samples": [0.5, 0.75], "bootstrap":[True, False]}

        tree_bayes = BayesSearchCV(trees_model, trees_params, n_iter=num_iter, cv=cv, random_state=101, n_jobs=-1, 
                                scoring="r2", return_train_score = True)
        
        tree_bayes.fit(X_train, y_train.values.reshape(len(y_train),))
        best_est = tree_bayes.best_estimator_
        HYPTERPARAM=True
        trial = False
    else: 

        tree_bayes = joblib.load(open(os.path.join("data", "augmented", "regression", "separate_datasets", "bagging_extra_fit", "new", f"bagging_extra_fit_model_{temp}.pkl"), "rb"))

        n_estimators = tree_bayes.get_params()["estimator"].n_estimators
        max_samples = tree_bayes.get_params()["estimator"].max_samples
        bootstrap = tree_bayes.get_params()["estimator"].bootstrap
        best_est = BaggingRegressor(ExtraTreeRegressor(),random_state=101, verbose=0, n_estimators=n_estimators, max_samples=max_samples, bootstrap=bootstrap)
        trees_params = { "n_estimators":n_estimators, "max_samples":max_samples, "bootstrap":bootstrap}
        HYPTERPARAM=False
        trial = True

    y_pred, _, model_train_score, result, width_average_score, width_average_trial = mapie_training(best_est,
                                                        X_train, y_train, generator, saving_model, 
                                                        temp, x_test, name, tree_bayes, trees_params, saving_data, data_type=data_type, 
                                                        HYPTERPARAM=HYPTERPARAM, y_test_trial = y_test_trial, x_test_trial=x_test_trial,
                                                        y_test_trial_top=y_test_trial_top, y_test_trial_bottom=y_test_trial_bottom, trial=trial)
    
    return y_pred, best_est, model_train_score, result, width_average_score, width_average_trial

