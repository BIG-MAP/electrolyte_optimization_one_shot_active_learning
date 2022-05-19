import os
import pandas as pd
import sys
import os.path
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from utils import *
import matplotlib as mpl
from database import Database

data = Database()
temperatures = data.sorted_temperatures

# maximum gradient calculation of the preducted conductivity according to each and both solvents
error_ratio_calc = pd.read_csv(os.path.join(os.getcwd(),"data", "augmented", "regression", "combined_datasets","fomulation_error", "max_gradient", "params_fomulation_error_all.csv"))
error_ratio_calc.set_index('temperature', inplace=True)

# median conductivtiy measurement error  data\augmented\regression\combined_datasets fomulation_error\exprtimental_error
error_range = pd.read_csv(os.path.join(os.getcwd(),"data", "augmented", "regression", "combined_datasets", "fomulation_error", "expertimental_error", "ranges.csv"),sep=";") 
error_range = error_range.rename(columns={'Unnamed: 0': 'temperature'})
error_range.set_index('temperature', inplace=True)
median_const = error_range.loc["total", 'medianRange']

error_ratio_calc["max_gradient_formulation_norm"] = median_const/error_ratio_calc["max_gradient_formulation_norm"]
error_ratio_calc["max_gradient_pc_norm"] = median_const/error_ratio_calc["max_gradient_pc_norm"]
error_ratio_calc["max_gradient_li_norm"] = median_const/error_ratio_calc["max_gradient_li_norm"]

plt.figure()
error_ratio_calc.plot(kind='bar', fontsize=8.5)
mf = mpl.ticker.ScalarFormatter(useMathText=True)
mf.set_powerlimits((0,0))
plt.gca().yaxis.set_major_formatter(mf)

plt.ylabel(r"$e_r$", fontdict=dict(size=13.5))
plt.xlabel("T [°C]", fontdict=dict(size=12.5))
plt.legend([r"$e_{r_{PC, LiPF_6}}$", r"$e_{r_{PC}}$", r"$e_{r_{LiPF_6}}$"])
plt.savefig(os.path.join("plots", "augmented", "regression", "ratio_error", f"fomulation_error_all_median_const_over_max_gradient_new.svg"))
plt.savefig(os.path.join("plots", "augmented", "regression", "ratio_error", f"fomulation_error_all_median_const_over_max_gradient_new.png"))
plt.clf()

