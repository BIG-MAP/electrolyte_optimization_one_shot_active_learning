# This file is used for interprating the solvent and salt interaction based on the extracted formula from the trained polynomial model
import os 
import re
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from ml_util import *
from database import Database

eq = json.load(open(os.path.join("data", "augmented", "regression", "combined_datasets", "poly_fit_optimized", "new","poly_fit_equation.json")))
data = Database()
temperatures = data.sorted_temperatures

plt.figure(figsize=(3.5,3.5))
for i in range(10):
    if i == 0:
        coeeficient = [(float(re.findall(r"\d+\.\d+", eq[f"poly_fit_{temp}"])[0]) + float(re.findall(r"\d+\.\d+", eq[f"poly_fit_{temp}"])[1])) for temp in temperatures]
    else: 
        coeeficient = [float(re.findall(r"\d+\.\d+", eq[f"poly_fit_{temp}"])[i+1]) for temp in temperatures]
    
    plt.plot(temperatures, coeeficient, label=fr"$c_{i}$", color= plt.get_cmap("tab10").colors[i])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) 
    plt.legend(fontsize=7, frameon=False)

plt.xticks([i for i in np.arange(-30.0, 60.0, 20.0)], rotation=90)
plt.ylabel("Coeeficients", fontdict=dict(size=12.5))
plt.xlabel("T [°C]", fontdict=dict(size=12.5))
plt.savefig("coefficient_effect.svg")
plt.savefig("coefficient_effect.png")



