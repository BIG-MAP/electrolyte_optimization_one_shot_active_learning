import warnings
import joblib
warnings.filterwarnings("ignore")
from sklearn.metrics import (
    plot_confusion_matrix,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    plot_roc_curve,
    f1_score)
import itertools
import seaborn as sns
import json
import numpy as np
import random
from numpy import busday_offset, sum as arraysum
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
from matplotlib import colors
import pandas as pd
import os
from pathlib import Path
from mapie.metrics import regression_coverage_score
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.rc('font', size=22)
matplotlib.rc('axes', titlesize=22)

plt.style.use(['nature', 'science', 'no-latex'])
plt.rcParams['text.usetex'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

data_dir = os.path.join(Path(__file__).parents[1].absolute(), "data")
plot_dir = os.path.join(Path(__file__).parents[1].absolute(), "plots")
original_dataset_name = "OriginalData.csv"

def load_dataset(dataset_type="original", method=None, aggregation=None, name=None, generator=None, data_type="old", seperator=False):
    """load dataset of interest

    Args:
        dataset_type (str, optional): type of the dataset. Defaults to "original".
        method (str, optional): type of supervised algorithm. Defaults to None.
        aggregation (bool, optional): Whether dataset is seperated for every temperature or not. Defaults to None.
        name (str, optional): name of the dataset of choice. Defaults to None.
        generator (str, optional): name of the regressor of choice. Defaults to None.
        data_type (str, optional): type of dataset (before and after one shot al). Defaults to "old".
        seperator (bool, optional): whether dataset belongs to an individual temperature or not . Defaults to False.

    Returns:
       dataframe: dataset of interest
    """

    if (dataset_type != "original") and (dataset_type != "augmented") and (dataset_type != "formulation_suggestions"):
        raise ValueError("No dataset with this type")
    if dataset_type == "original":
        if name and seperator:
            dataset = pd.read_csv(os.path.join(
                data_dir, dataset_type, name), sep=";")
        elif name and not seperator:
            dataset = pd.read_csv(os.path.join(
                data_dir, dataset_type, name))
        else:
            dataset = pd.read_csv(os.path.join(
                data_dir, dataset_type, original_dataset_name))
        if "Unnamed: 0" in dataset:
            del dataset["Unnamed: 0"]
        return dataset
    if dataset_type == "augmented":

        if name == None:
            raise ValueError("No database name given")
        if aggregation == None:
            raise ValueError("No aggregation level given")
        if method == None:
            raise ValueError(
                "No method level given. Should be either classification or augmentation.")
        if aggregation:
            dataset = os.path.join(
                data_dir, dataset_type, method, "combined_datasets", generator, name)
        elif data_type:
            dataset = os.path.join(
                data_dir, dataset_type, method, "separate_datasets", generator, data_type, name)
        else:
            dataset = os.path.join(
                data_dir, dataset_type, method, "separate_datasets", generator, name)
        dataset = pd.read_csv(dataset)
        if "Unnamed: 0" in dataset:
            del dataset["Unnamed: 0"]
            return dataset
    if dataset_type == "formulation_suggestions":
        dataset = pd.read_csv(os.path.join(data_dir, dataset_type, name))
        return dataset
        


def save_directory(augmentation_method="regression", generator="ctgan", separate=None):
    """save the dataset in the defined directory 

    Args:
        augmentation_method (str, optional): type of supervised method. Defaults to "regression".
        generator (str, optional): name of the estimator of the choice. Defaults to "ctgan".
        separate (bool, optional): whether dataset belongs to a specific temperature or not. Defaults to None.

    Returns:
        str: create directory of the interest for saving the dataset
    """

    if separate:
        save_dir = os.path.join(
            data_dir, "augmented", augmentation_method, "separate_datasets", generator)
    else:
        save_dir = os.path.join(
            data_dir, "augmented", augmentation_method, "combined_datasets", generator)

    check_folder = os.path.isdir(save_dir)
    if not check_folder:
        os.makedirs(save_dir)
    return save_dir


def save_dataset(dataset, augmentation_method="regression", generator="ctgan", data_type=None, separate=None, temperature=None, name=False):
    """method for saving the dataset

    Args:
        dataset (dataframe): dataset of interest
        augmentation_method (str, optional): type of supervised method. Defaults to "regression".
        generator (str, optional): name of the estimator of the choice. Defaults to "ctgan".
        data_type (str, optional): type of dataset of interest (before and after one shot al). Defaults to None.
        separate (book, optional): whether dataset belongs to a specific temperature or not. Defaults to None.
        temperature (float, optional): temperature of choice. Defaults to None.
        name (bool, optional): name that dataset should be saved under it. Defaults to False.
    """
    if separate:
        save_dir = save_directory(augmentation_method, generator, separate)
    else:
        save_dir = save_directory(augmentation_method, generator, separate)
        temperature = "all"

    check_folder = os.path.isdir(save_dir)
    if not check_folder:
        os.makedirs(save_dir)

    if data_type:
        path_data_type = os.path.join(save_dir, data_type)
        check_folder = os.path.isdir(path_data_type)
        if not check_folder:
            os.makedirs(path_data_type)
        if name:

            dataset.to_csv(os.path.join(path_data_type, f"params_{generator}_{temperature}.csv"))
        else:
            dataset.to_csv(os.path.join(path_data_type,  f"{generator}_{temperature}.csv"))
    else:
        if name:

            dataset.to_csv(os.path.join(save_dir, f"params_{generator}_{temperature}.csv"))
        else:
            dataset.to_csv(os.path.join(save_dir, f"{generator}_{temperature}.csv"))

def save_model(model, augmentation_method="regression", generator="ctgan", separate=None, temperature=None, dumping=False, data_type=None):
    """save the trained model

    Args:
        model: trained and fitted model
        augmentation_method (str, optional): type of supervised method. Defaults to "regression".
        generator (str, optional): name of the estimator of the choice. Defaults to "ctgan".
        separate (book, optional): whether dataset belongs to a specific temperature or not. Defaults to None.
        temperature (float, optional): temperature of choice. Defaults to None.
        dumping (bool, optional): whether the model should be dumped or not. Defaults to False.
        data_type (str, optional): type of dataset of interest (before and after one shot al). Defaults to None.
    """

    save_dir = save_directory(augmentation_method, generator, separate)
    if data_type:
        data_type_dir = os.path.join(save_dir, data_type)
        if not os.path.isdir(data_type_dir):
            os.makedirs(data_type_dir)

        name =  os.path.join(data_type_dir, f'{generator}_model_{temperature}.pkl')
    else: 
        name =  os.path.join(save_dir, f'{generator}_model_{temperature}.pkl')
    
    if dumping:
        joblib.dump(model,name)
    else:
        model.save(name)

def save_result(data, separate=None):
    """creating the path for saving the result 

    Args:
        data (dataframe): dataset of the interest
        separate (book, optional): whether dataset belongs to a specific temperature or not. Defaults to None.
    """
    if separate:
        file_name = "separate"
    else:
        file_name = "combined"

    data.to_csv(os.path.join(data_dir, "augmented",
                f"results_{file_name}.csv"))


def plot_material(x, y, colour, x_label="Pc/(Ec+Pc)", y_label="LiPF6/(Ec+Pc)", title=None, file_name="scatter_plot", dataset_type="original",
                  augmentation_method="regression", annotate_point=False, ann=None, point_size=None, generator=False, figsize=(10, 10),
                  contour=False, square=False, colorbar=True, real_data=False, x_real=None, y_real=None, 
                  plot=False, y_prime=None, xticks_range=[0.09, 0.2, 0.4, 0.6, 0.8, 1], retrain=False, contour_level=10, 
                  contour_level_line=5, colorbar_range=False,
                  colorbar_range_min=-30.0, colorbar_range_max=60.0,contourname=False, data_type="new", uncertaiinty=False, ROTATION=False):

    """contour plot for visualising the results

    Args:
        x (array): data for x-axis
        y (array): data for y-axis
        colour (array): the data which colorbar will be created according to that.
        x_label (str, optional): x-axis label. Defaults to "Pc/(Ec+Pc)".
        y_label (str, optional): y_axis label. Defaults to "LiPF6/(Ec+Pc)".
        title (str, optional): plot title. Defaults to None.
        file_name (str, optional): name of the file. Defaults to "scatter_plot".
        dataset_type (str, optional): type of the dataset. Defaults to "original".
        augmentation_method (str, optional): type of supervised algorithm. Defaults to "regression".
        annotate_point (bool, optional): whether the points should be annotated or not. Defaults to False.
        ann (array, optional): data used for annotation. Defaults to None.
        point_size (int, optional): size of data points. Defaults to None.
        generator (bool, optional): name of the generator. Defaults to False.
        figsize (tuple, optional): size of the figure. Defaults to (10, 10).
        contour (bool, optional): whether the plot should be a 2d contour or not. Defaults to False.
        square (bool, optional): whether the plot should be squared or not. Defaults to False.
        colorbar (bool, optional): whether colorbar should be drawn or not. Defaults to True.
        real_data (bool, optional): whether real data should be also in the plot or not. Defaults to False.
        x_real (array, optional): real measurement for x axis. Defaults to None.
        y_real (array, optional): real measurement for y axis. Defaults to None.
        plot (bool, optional): whether line plot should be drawn or not. Defaults to False.
        y_prime (array, optional): the output of the testset for line plot. Defaults to None.
        xticks_range (list, optional): the ticks range of the x axis. Defaults to [0.09, 0.2, 0.4, 0.6, 0.8, 1].
        retrain (bool, optional): whether the data belongs to post learning or not. Defaults to False.
        contour_level (int, optional): number of contour lines. Defaults to 10.
        contour_level_line (int, optional): number of contous lines. Defaults to 5.
        colorbar_range (bool, optional): the range of the colorbar. Defaults to False.
        colorbar_range_min (float, optional): min range of the colorbar. Defaults to -30.0.
        colorbar_range_max (float, optional): max range of the colorbar. Defaults to 60.0.
        contourname (bool, optional): name of the contout plto. Defaults to False.
        data_type (str, optional): dataset type. Defaults to "new".
        uncertaiinty (bool, optional): whether data belongs to uncertainty measurements or not. Defaults to False.
        ROTATION (bool, optional): whether axis ticks should be rotated or not. Defaults to False.
    """

    # check if there is a folder, if not create it
    plot_save_dir = os.path.join(plot_dir, dataset_type, augmentation_method)
    check_folder = os.path.isdir(plot_save_dir)
    if not check_folder:
        os.makedirs(plot_save_dir)

    if not isinstance(colour, pd.Series):
        colour = pd.Series(colour)

    fig = plt.figure(figsize=figsize)
    if contour:
        plt.style.use(['nature', 'science', 'no-latex'])
        sc1 = plt.contour(x.values.reshape(int(np.sqrt(len(x))), int(np.sqrt(len(x)))),
                    y.values.reshape(int(np.sqrt(len(y))),
                                     int(np.sqrt(len(y)))),
                    colour.values.reshape(int(np.sqrt(len(x))), int(np.sqrt(len(y)))), colors='darkgrey', levels=contour_level_line)

        sc = plt.contourf(x.values.reshape(int(np.sqrt(len(x))), int(np.sqrt(len(x)))),
                          y.values.reshape(int(np.sqrt(len(y))),
                                           int(np.sqrt(len(y)))),
                          colour.values.reshape(int(np.sqrt(len(x))), int(np.sqrt(len(y)))),
                          levels=contour_level)
        plt.clabel(sc1, inline=False, colors="k", fontsize=10)

        if colorbar:
            if uncertaiinty:
                # Colormap 
                cmap = plt.get_cmap('viridis', 18)
                # Normalizer 
                norm = mpl.colors.Normalize(vmin=0, vmax=6)
                
                # creating ScalarMappable 
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([]) 
                
                #plt.clim(0, 0.006)
                cb = plt.colorbar(sm, ticks=np.linspace(0, 6, 6), shrink=1, pad=-1.23, orientation="horizontal")
                cb.formatter.set_powerlimits((0, 0))
                cb.update_ticks()
                cb.ax.tick_params(labelsize=12, rotation=90)
                
            elif colorbar_range:
                plt.clim(colorbar_range_min, colorbar_range_max)
                cb = plt.colorbar(sc, shrink=1, pad=-1.12, orientation="horizontal")
                cb.formatter.set_powerlimits((0, 0))
                cb.update_ticks()

            else:
                cb = plt.colorbar(sc, shrink=1, pad=-1.23, orientation="horizontal")
                cb.set_label(label="",size=12)
                cb.ax.tick_params(labelsize=12, rotation=90)
                cb.formatter.set_powerlimits((0,0))
                cb.update_ticks()

    else:
        if point_size:
            sc = plt.scatter(x, y, c=colour, s=point_size, alpha=0.6)
        else:
            sc = plt.scatter(x, y, c=colour)
        if colorbar:
            if colorbar_range:
                plt.clim(colorbar_range_min, colorbar_range_max)
            else:pass

            cb = plt.colorbar(sc)
            cb.formatter.set_powerlimits((0, 0))
            cb.update_ticks()

    if real_data:
        plt.scatter(x_real, y_real, c="orange", marker="x")
    if annotate_point:
        for i in range(len(x)):
            plt.annotate(ann.values[i], (x.values[i],  y.values[i]),
                         fontsize=6, ha='center', va='center')

    if plot:
        plt.plot(x, y_prime)
    if title:
        plt.title(title, pad=10)

    if ROTATION:
        plt.xticks(xticks_range, rotation=90, fontsize=12)
    else:
        plt.xticks(xticks_range, fontsize=12)
    
    plt.yticks(fontsize=12)
    # subscript plot
    plt.rcParams.update({'mathtext.default':  'regular' })
    plt.xlabel(x_label, fontsize=14.5)
    plt.ylabel(y_label, fontsize=14.5)
    plt.xlim(min(xticks_range), max(xticks_range))
    plt.ylim(0.1,1.1,0.2)
    if square:
        plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    if retrain:
        save_path= os.path.join(plot_save_dir,  generator, "contour", data_type)
        check_folder = os.path.isdir(save_path)    
        if not check_folder:
            os.makedirs(save_path)    
        plt.savefig(os.path.join(save_path, f"{file_name}.png"))
        plt.savefig(os.path.join(save_path, f"{file_name}.svg"))
    elif contourname:
        save_path= os.path.join(plot_save_dir,  generator, "confidance_interval", data_type)
        check_folder = os.path.isdir(save_path)    
        if not check_folder:
            os.makedirs(save_path)    
        plt.savefig(os.path.join(save_path, f"{file_name}.png"))
        plt.savefig(os.path.join(save_path, f"{file_name}.svg"))
    elif not generator:
        plt.savefig(os.path.join(plot_save_dir, "generated_data", f"{file_name}.svg"))
        plt.savefig(os.path.join(plot_save_dir, "generated_data", f"{file_name}.png"))
    else:
        plt.savefig(os.path.join(plot_save_dir, generator, f"{file_name}.svg"))
        plt.savefig(os.path.join(plot_save_dir, generator, f"{file_name}.png"))
    plt.clf()


def plot_highest_cond_plot(generator, dataset, temperatures,  file_name, data_type, top_folder="top_perc", fsize=9, x_label="Pc/(Ec+Pc)", y_label="LiPF6/(Ec+Pc)",
                          dataset_type="augmented", augmentation_method="regression", annotate_point=True, xticks_range=[0.09, 0.2, 0.4, 0.6, 0.8, 1], ZOOM=False):
    """plot function for drawing the top measuremets 

    Args:
        generator (str, optional): name of the estimator of the choice.
        dataset (dataframe): dataset of interest
        temperatures (float): temperature of interest
        file_name (str): name of the file
        data_type (str): dataset type.
        top_folder (str, optional): name of the folder which results need to be saved. Defaults to "top_perc".
        fsize (int, optional): figure size. Defaults to 9.
        x_label (str, optional): label of x-axis. Defaults to "Pc/(Ec+Pc)".
        y_label (str, optional): label of y-axis. Defaults to "LiPF6/(Ec+Pc)".
        dataset_type (str, optional): type of the dataset. Defaults to "augmented".
        augmentation_method (str, optional): type of the supervised algorithm. Defaults to "regression".
        annotate_point (bool, optional): if annotation is required or not. Defaults to True.
        xticks_range (list, optional): ticks of x axis ticks. Defaults to [0.09, 0.2, 0.4, 0.6, 0.8, 1].
        ZOOM (bool, optional): whether zoom is required or not. Defaults to False.
    """
    # check if there is a folder, if not create it
    plot_save_dir = os.path.join(plot_dir, dataset_type, augmentation_method)
    check_folder = os.path.isdir(plot_save_dir)
    if not check_folder:
        os.makedirs(plot_save_dir)
    plt.style.use(['nature', 'science', 'no-latex'])
    plt.figure(figsize=(fsize,fsize))
    x_data, y_data, x_err_l, y_err_l, x_err_h, y_err_h, temp_data, color_data = [], [], [], [], [], [], [], []
    for temp in temperatures:
        temp_data.append(temp)
        x_data.append(np.mean(dataset["PcEcPc"][dataset["temperature"]==temp]))
        y_data.append(np.mean(dataset["liEcPc"][dataset["temperature"]==temp]))
        x_err_l.append(np.min(dataset["PcEcPc"][dataset["temperature"]==temp]))
        y_err_l.append(np.min(dataset["liEcPc"][dataset["temperature"]==temp]))
        x_err_h.append(np.max(dataset["PcEcPc"][dataset["temperature"]==temp]))
        y_err_h.append(np.max(dataset["liEcPc"][dataset["temperature"]==temp]))
        color_data.append(np.max(dataset["conductivity"][dataset["temperature"]==temp]))
    
    ## adding axis for zooming later
    fig, ax = plt.subplots(figsize=[fsize+0.3, fsize])
    
    ax.plot(x_data, y_data, linestyle='', color="k")
    sc= ax.scatter(x=x_data, y=y_data, c=color_data, s=15)
    er = ax.errorbar(x=np.array(x_data), y=np.array(y_data), xerr=(np.array(x_data) - np.array(x_err_l), np.array(x_err_h) - np.array(x_data)),
                 yerr=(np.array(y_data) - np.array(y_err_l), np.array(y_err_h) - np.array(y_data)),
                 fmt="none", zorder=0, c="k",capsize=4, elinewidth=1, linestyle="--") # marker='o', yerr=(y_err_l, y_err_h),#, marker =''
    er[-1][0].set_linestyle('--')
    er[-1][1].set_linestyle('--')
    xticks_range = [i for i in np.arange(0, 1.3, 0.2)]
    yticks_range= [i for i in np.arange(0, 1.3, 0.2)]
    
    ax.set_xlim(min(xticks_range), max(xticks_range))
    ax.set_ylim(min(yticks_range), max(yticks_range))
    ax.set_xticks(xticks_range)
    ax.set_yticks(yticks_range)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    cb =fig.colorbar(sc, ax=ax, ticks=np.linspace(0, 0.016, 8), shrink=1, orientation="vertical" )
    cb.set_label(label="",size=20)
    cb.set_ticks(np.arange(np.min(dataset["conductivity"]),np.max(dataset["conductivity"])+0.00100,0.00400))
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()
    
    axins = ax.inset_axes([0.6, 0.6, 0.38, 0.38])
    axins.plot(x_data, y_data, '--', color='k')
    axins.scatter(x=x_data, y=y_data, c=color_data, s=60)

    if annotate_point:
        for i, temp in enumerate(temp_data):
            axins.annotate(str(temp), xy=(x_data[i]+0.040,  y_data[i]-0.005),
                         fontsize=5, ha='center', va='center', c="k", weight="bold")

    axins.set_xlim(min(x_err_l)-0.0075, max(x_err_h)+0.0075)
    axins.set_ylim(min(y_err_l)-0.0065, max(y_err_h)+0.0065)
    axins.set_yticklabels([])
    axins.set_xticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="black")
    if not top_folder:
        plt.savefig(os.path.join(plot_save_dir, generator, f"{file_name}.svg"))
        plt.savefig(os.path.join(plot_save_dir, generator, f"{file_name}.png"))
    else:
        save_path= os.path.join(plot_save_dir,  generator, top_folder, data_type)
        check_folder = os.path.isdir(save_path)    
        if not check_folder:
            os.makedirs(save_path)    
        plt.savefig(os.path.join(save_path, f"{file_name}.svg"))
        plt.savefig(os.path.join(save_path, f"{file_name}.png"))
    plt.clf()


def kde_plot(data, temperatures=None, hue="temperature", x="PcEcPc", y="liEcPc", aggregation=True,
             augmentation_method="regression", dataset_type="augmented", generator="tvae"):

    plot_save_dir = os.path.join(
        plot_dir, dataset_type, augmentation_method, generator, "kde")
    check_folder = os.path.isdir(plot_save_dir)
    if not check_folder:
        os.makedirs(plot_save_dir)
    if aggregation:
        gfg = sns.jointplot(data=data, x=x, y=y, hue=hue,
                            s=12, height=8, palette="tab10")
        gfg.ax_marg_x.set_xlim(0, 1.01)
        gfg.ax_marg_y.set_ylim(0, 1.2)
        legend_properties = {'size': 8}
        gfg.ax_joint.legend(title='Temperature',
                            prop=legend_properties, loc='upper left')
        plt.savefig(os.path.join(plot_save_dir, f"kde_{generator}_total.png"))
        plt.clf()
    else:
        palette = itertools.cycle(sns.color_palette("tab10"))
        for temp in temperatures:
            tmp_data = data[data["temperature"] == temp]
            gfg = sns.jointplot(data=tmp_data, x=x, y=y,
                                height=8, color=next(palette), kind="reg")
            gfg.ax_marg_x.set_xlim(0, 1.01)
            gfg.ax_marg_y.set_ylim(0, 1.2)
            legend_properties = {'size': 8}
            plt.savefig(os.path.join(plot_save_dir,
                        f"kde_{generator}_{temp}.png"))
            plt.clf()
    return


def plot_regression_results(ax, y_true, y_pred, title, scores):
    """Scatter plot of the predicted vs true targets."""
    ax.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--r", linewidth=2
    )
    ax.scatter(y_true, y_pred, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))
    #ax.set_xlim([y_true.min(), y_true.max()])
    #ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    extra = plt.Rectangle(
        (0, 0), 0, 0, fc="w", fill=False, edgecolor="none", linewidth=0
    )
    ax.legend([extra], [scores], loc="upper left")
    ax.set_title(title)


def data_balance(dataset, axsis_name, file_name, dataset_type="original"):
    fig, ax1 = plt.subplots(figsize=(20, 10))
    graph = sns.countplot(ax=ax1, x=axsis_name, data=dataset)
    graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
    i = 0
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 0.1,
                   dataset[axsis_name].value_counts()[i], ha="center")
        i += 1
    plt.savefig(os.path.join(plot_dir, dataset_type, file_name))


def metric_calc(y_test, predict_normal, name):

    tn, fp, fn, tp = confusion_matrix(y_test, predict_normal).ravel()
    # f1 score
    f1 = f1_score(y_test, predict_normal)
    # precision recall
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    # specificity or true negative rate
    specificity = tn / (tn + fp)
    # sensitivity or true positive rate
    sensitivity = tp / (tp + fn)
    # false positive rate
    fpr = fp / (fp + tn)
    # overall accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    print(
        f"precision: {precision} \n recall {recall} \n f1 {f1} \n accuracy {acc}")
    with open(os.path.join(data_dir, "augmented", "classification", "combined_datasets", "ada", name), 'w') as f:
        f.write(
            f"precision: {precision}\n recall {recall}\n f1 {f1}\n accuracy {acc}")


def save_top_result(data, name, generator, augmentation_method="regression", separate=None, data_type=None):
    if separate:
        file_name = "separate"
        sub_dir = "separate_datasets"
    else:
        file_name = "combined"
        sub_dir = "combined_datasets"

    if data_type:
        data.to_csv(os.path.join(data_dir, "augmented", augmentation_method,
                sub_dir, generator, data_type,f"top_results_{file_name}_{name}.csv"))
    else:
        data.to_csv(os.path.join(data_dir, "augmented", augmentation_method,
                sub_dir, generator, f"top_results_{file_name}_{name}.csv"))


def helper_calc(y_test, y_pred_original, x_test, temp, min_deg, generator, PLOT=False):
    if not y_pred_original.ndim == 2:
        y_pred_original = y_pred_original.reshape(y_pred_original.shape[0], 1)
    sum_errs = arraysum((y_test - y_pred_original)**2)
    stdev = np.sqrt(1/(len(y_test)-2)*sum_errs)
    interval = 1.96*stdev

    upper_pred_interval, lower_pred_interval = [], []
    for point in y_pred_original:
        lower, upper = point-interval, point+interval
        upper_pred_interval.append(upper)
        lower_pred_interval.append(lower)

    if PLOT:
        plt.figure()
        y_test_fill = np.sort(y_test.iloc[:, 0])
        y_test_fill_index = np.argsort(y_test.iloc[:, 0]).tolist()
        y_1_fill = np.array(upper_pred_interval).flatten()[y_test_fill_index]
        y_2_fill = np.array(lower_pred_interval).flatten()[y_test_fill_index]
        y_pred_original_sort = y_pred_original[y_test_fill_index]

        plt.scatter(y_test_fill, y_pred_original_sort.flatten())
        plt.fill_between(y_test_fill, y_1_fill, y_2_fill, alpha=0.4, color="m")
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plot_save_dir_confidance = os.path.join(
            plot_dir, "augmented", "regression", "confidance")
        if not os.path.isdir(plot_save_dir_confidance):
            os.makedirs(plot_save_dir_confidance)
        plt.savefig(os.path.join(plot_save_dir_confidance,
                    f"comparison_at_temp_{temp}_{generator}.png"))
        plt.clf()

    mae = mean_absolute_error(y_test, y_pred_original)
    r2 = r2_score(y_test, y_pred_original)
    if PLOT:
        fig, axs = plt.subplots(1, 1, figsize=(9, 9))
        #axs = np.ravel(axs)
        scores = (r"R^2={:.2f}" + "\n" + r"MAE={:.5f} ").format(r2, mae)
        if generator == "poly":
            title = f"{temp} with degree {min_deg} and {generator}"
        else:
            title = f"{temp} with {generator}"

        plot_regression_results(axs, y_test, y_pred_original,
                                title=title, scores=scores)
        

        plot_save_dir = os.path.join(
            plot_dir, "augmented", "regression", "poly_analysis")
        fig.tight_layout()
        if not os.path.isdir(plot_save_dir):
            os.makedirs(plot_save_dir)

        plt.savefig(os.path.join(plot_save_dir,
                    f"poly_confidance_{temp}_{generator}.png"))
        plt.clf()


def uncertainty_plot_poly(y_test, y_predict, n_repeat, name, y_error, y_bias, y_noise, y_var, n_estimators, temp):

    plt.subplot(3, n_estimators, 1)
    y_test_plot = np.sort(y_test.iloc[:, 0])
    y_test_plot_index = np.argsort(y_test.iloc[:, 0]).tolist()

    for i in range(n_repeat):
        if i == 0:
            plt.scatter(
                y_test_plot, y_predict[:, i][y_test_plot_index], color="r", label=r"$\^y$")
        else:
            plt.scatter(
                y_test_plot, y_predict[:, i][y_test_plot_index], color="r", alpha=0.05)

    plt.scatter(y_test, np.mean(y_predict, axis=1),
                color="c", label=r"$\mathbb{E}_{LS} \^y(x)$")

    plt.title(f"{name} at {temp}")
    plt.legend(loc=(1.1, 0.5))
    plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    plt.subplot(3, n_estimators, n_estimators + 1)
    plt.plot(y_test_plot, y_bias, color="b", label="$bias^2(x)$"),
    plt.plot(y_test_plot, y_var, "g", label="$variance(x)$"),
    plt.legend(loc=(1.1, 0.5))
    plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    plt.subplot(3, n_estimators, n_estimators + 2)
    plt.plot(y_test_plot, y_error, "r", label="$error(x)$")
    plt.plot(y_test_plot, y_noise, "c", label="$noise(x)$")
    plt.legend(loc=(1.1, 0.5))
    plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    plt.subplots_adjust(right=0.75)
    plot_save_dir = os.path.join(
        plot_dir, "augmented", "regression", "uncertainty")
    if not os.path.isdir(plot_save_dir):
        os.makedirs(plot_save_dir)  
    
    plt.savefig(os.path.join(plot_save_dir, f"{name}_{temp}.png"))
    plt.clf()


def poly_lin_fit(x, y, x_test, degree):
    # Train features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly_train = poly_features.fit_transform(x)

    # Linear regression
    poly_reg = LinearRegression()
    poly_reg.fit(x_poly_train, y)

    # Compare with test data
    x_poly_test = poly_features.fit_transform(x_test)
    poly_predict = poly_reg.predict(x_poly_test)
    pol_score = poly_reg.score(x_poly_train, y)
    return pol_score, poly_predict


def find_percentile(data):
    data["percentile_rank"] = data.conductivity.rank(pct=True)
    return data


def cross_poly_calc(x, y, degree, num_cross_vals, temp, generator, file_name, dataset_type="original", augmentation_method="regression"):
    plot_save_dir = os.path.join(plot_dir, dataset_type, augmentation_method)
    check_folder = os.path.isdir(plot_save_dir)
    if not check_folder:
        os.makedirs(plot_save_dir)

    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1e-4))
    scores = cross_val_score(
        model, x, y, cv=num_cross_vals, scoring="neg_mean_squared_error")
    plt.plot([i for i in range(num_cross_vals)], abs(scores))
    plt.xlabel("# of Folds")
    plt.ylabel("MSE")
    plt.title(f"{generator} at {temp}° ")
    plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    if not generator:
        path_folder = os.path.join(plot_save_dir, "generated_data")
        check_folder = os.path.isdir(path_folder)    
    else:
        path_folder = os.path.join(plot_save_dir, generator)
        check_folder = os.path.isdir(path_folder)

    if not check_folder:
            os.makedirs(path_folder)  
    plt.savefig(os.path.join(path_folder, file_name))
    plt.clf()
    return scores


def violin_plot_two_set(data, temp, generator, file_name, dataset_type="augmented", augmentation_method="regression"):
    plot_save_dir = os.path.join(plot_dir, dataset_type, augmentation_method)
    check_folder = os.path.isdir(plot_save_dir)
    if not check_folder:
        os.makedirs(plot_save_dir)
    fig = plt.figure(figsize=(7, 7))
    sns.violinplot(x="temperature", y="conductivity", hue="kind",
                   data=data, split=True, scale="count")
    plt.title(f"temp {temp}")
    sns.despine()

    plt.tight_layout()

    if not generator:
        plt.savefig(os.path.join(plot_save_dir, "generated_data", file_name))
    else:
        plt.savefig(os.path.join(plot_save_dir, generator, file_name))
    plt.clf()


def categorical_kde_plot(df, variable, category, category_order=None, horizontal=False, rug=True, figsize=None):
    """Draw a categorical KDE plot

    Parameters
    ----------
    df: pd.DataFrame
        The data to plot
    variable: str
        The column in the `df` to plot (continuous variable)
    category: str
        The column in the `df` to use for grouping (categorical variable)
    horizontal: bool
        If True, draw density plots horizontally. Otherwise, draw them
        vertically.
    rug: bool
        If True, add also a sns.rugplot.
    figsize: tuple or None
        If None, use default figsize of (7, 1*len(categories))
        If tuple, use that figsize. Given to plt.subplots as an argument.
    """
    if category_order is None:
        categories = list(df[category].unique())
    else:
        categories = category_order[:]

    figsize = (7, 1.0 * len(categories))
    fig, axes = plt.subplots(
        nrows=len(categories) if horizontal else 1,
        ncols=1 if horizontal else len(categories),
        figsize=figsize[::-1] if not horizontal else figsize,
        sharex=horizontal,
        sharey=not horizontal)

    for i, (cat, ax) in enumerate(zip(categories, axes)):
        sns.kdeplot(
            data=df[df[category] == cat],
            x=variable if horizontal else None,
            y=None if horizontal else variable,
            bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5, ax=ax, color="lightslategray")

        keep_variable_axis = (i == len(fig.axes) -1) if horizontal else (i == 0)

        if rug:
            sns.rugplot(
                data=df[df[category] == cat],
                x=variable if horizontal else None,
                y=None if horizontal else variable,
                ax=ax, color="black", height=0.025 if keep_variable_axis else 0.04)
        _format_axis(ax, cat, horizontal, keep_variable_axis=keep_variable_axis,)
    plt.tight_layout()
    plt.show()


def _format_axis(ax, category, horizontal=False, keep_variable_axis=True):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if horizontal:
        ax.set_ylabel(None)
        lim = ax.get_ylim()
        ax.set_yticks([(lim[0] + lim[1]) / 2])
        ax.set_yticklabels([category])
        if not keep_variable_axis:
            ax.get_xaxis().set_visible(False)
            ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel(None)
        lim = ax.get_xlim()
        ax.set_xticks([(lim[0] + lim[1]) / 2])
        ax.set_xticklabels([category])
        if not keep_variable_axis:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)


def plot_3d_data(x_train, y_train, x_test, y_test, y_pred, y_pred_low, y_pred_up, ax=None, title=None):

    x_t = x_test[["PcEcPc"]].values.reshape((len(x_test[["PcEcPc"]]),))
    y_t = x_test[["liEcPc"]].values.reshape((len(x_test[["liEcPc"]]),))
    z_t = y_test.values.reshape((len(y_test),))
    ax = plt.axes(projection="3d")
    ax.scatter(x_train[["PcEcPc"]], x_train[["liEcPc"]],
               y_train, label="Training data")
    ax.scatter(x_t, y_t, z_t, label="Testing data")

    for i in range(len(y_test)):
        ax.plot([x_t[i], x_t[i]], [y_t[i], y_t[i]], [y_pred[i] +
                y_pred_up[i], y_pred[i]-y_pred_low[i]], marker="_", alpha=0.7)

    ax.set_xlabel("PcEcPc", fontsize=10)
    ax.set_ylabel("LiEcPc", fontsize=10)
    ax.set_zlabel(r"Conductivity [$\frac{1}{\Omega cm}$]", fontsize=10)
    ax.tick_params(axis='both', labelsize=10)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(False)
    plt.savefig(os.path.join("plots", "comparison",
                "uncertainty_pretrained", f"{title}.png"))
    plt.clf()


def converage_table(STRATEGIES, y_test, y_pis, temp):
    """coverage table creation 

    Args:
        STRATEGIES (str): strategy type
        y_test (array): label of the testset
        y_pis (array): prediction interval
        temp (float): temperature of the choice
    """
    converage_score_table = pd.DataFrame([[regression_coverage_score(
        y_test, y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]),
        (y_pis[strategy][:, 1, 0] - y_pis[strategy][:, 0, 0]).mean()] for strategy in STRATEGIES],
        index=STRATEGIES, columns=["Coverage", "Width average"]).round(3)
    converage_score_table.to_csv(os.path.join("plots", "comparison", "coverage",
                                              f"results_{temp}.csv"))


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def three_axes_plot(data_host, label_host, data_par, label_par, label_x, file_name, dataset_type="augmented",
                    augmentation_method="regression", square=True, generator="lazy"):
    plot_save_dir = os.path.join(plot_dir, dataset_type, augmentation_method)
    check_folder = os.path.isdir(plot_save_dir)
    if not check_folder:
        os.makedirs(plot_save_dir)
    fig, host = plt.subplots(figsize=(8, 8))
    par1 = host.twinx()
    p1, = host.plot(data_host, '--bo', label=label_host)
    p2, = par1.plot(data_par, '--go', label=label_par)

    host.set_ylim(min(data_host)-0.01, max(data_host)+0.01)
    par1.set_ylim(min(data_par)-0.0001, max(data_par)+0.0001)

    host.set_xlabel(label_x, fontdict=dict(size=16)) 
    host.set_ylabel(label_host, fontdict=dict(size=16)) 
    par1.set_ylabel(label_par, fontdict=dict(size=16))

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=15, **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), labelsize=15, **tkw)
    host.tick_params(axis='x', rotation=90, labelsize=12, **tkw)

    lines = [p1, p2]

    host.legend(lines, [l.get_label() for l in lines])
    if square:
        plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    if not generator:
        plt.savefig(os.path.join(plot_save_dir, "generated_data", file_name))
    else:
        plt.savefig(os.path.join(plot_save_dir, generator, file_name))
    plt.clf()

# catch the wanings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

