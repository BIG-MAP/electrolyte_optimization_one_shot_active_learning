import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ml_util import *
from database import Database

data = Database()
old_df = data._initial_data
new_df = data._updated_data

temperatures = data.sorted_temperatures
data_type, regressor = "new", "poly_fit"
plt.figure(figsize=(3,3))
for i, temp in enumerate(temperatures):

    data = pd.read_csv(os.path.join(os.getcwd(), "data", "augmented", "regression","separate_datasets",  f"{regressor}_optimized", data_type, f"{regressor}_{temp}.csv"))
    salt_predict = (1/151.905)*0.3*data["liEcPc"]
    salt_real = (1/151.905)*0.3*new_df["liEcPc"]
    plt.plot(salt_predict[::100], data["conductivity"][::100], label=temp, color= plt.get_cmap("tab10").colors[i])
    plt.scatter(salt_real[new_df["temperature"]==temp][::4], new_df["conductivity"][new_df["temperature"]==temp][::4], s=2, color= plt.get_cmap("tab10").colors[i])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-2,2)) 
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-2,2)) 
    plt.legend(fontsize=4)

plt.xlabel(r"$\frac{LiPF_{6}}{EC + PC + EMC}$ [$\frac{mol}{Kg}$]", fontdict=dict(size=12.5))
plt.ylabel(r"$\sigma$ [$\frac{S}{cm}$]", fontdict=dict(size=12.5))
plt.show()
