# This file can be used for finding the maximum gradient norm of predicted conductivity depending to different formulations.
import os
import json
from sympy import Symbol, Derivative
import pandas as pd
import numpy as np
import sys
import os.path
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
from utils import *
from database import Database

data = Database()
temperatures = data.sorted_temperatures
eq = json.load(open(os.path.join("data", "augmented", "regression", "combined_datasets","poly_fit_optimized", "new", "poly_fit_equation.json")))
data_type, regressor = "new", "poly_fit"
fidelity, conductivity_threshold = pd.DataFrame(), pd.DataFrame()

for j, temp in tqdm(enumerate(temperatures)):
    x0 = Symbol("x0")
    x1 = Symbol("x1")
    y = eval(eq[f"poly_fit_{temp}"][4:])
    p1 = Derivative(y,x0)
    p2 = Derivative(y,x1)
    
    data = pd.read_csv(os.path.join(os.getcwd(), "data", "augmented", "regression","separate_datasets", f"{regressor}_optimized", data_type, f"{regressor}_{temp}.csv"))
    gradient_formulation_norm, gradient_pc_norm, gradient_li_norm = [], [], []
    for i in range(len(data["PcEcPc"])):
        dp1_dx0= p1.subs({x0: data["PcEcPc"][i], x1: data["liEcPc"][i]}).doit()
        dp2_dx1 = p2.subs({x0: data["PcEcPc"][i], x1: data["liEcPc"][i]}).doit()
        gradient_calc = np.matrix([[float(dp1_dx0), float(dp2_dx1)]])
        gradient_formulation_norm.append(np.linalg.norm(gradient_calc))
        gradient_pc_norm.append(np.linalg.norm(float(dp1_dx0)))
        gradient_li_norm.append(np.linalg.norm(float(dp2_dx1)))

    fidelity = pd.concat([fidelity, pd.DataFrame([temp, gradient_formulation_norm, gradient_pc_norm, gradient_li_norm]).T])
    conductivity_threshold = pd.concat([conductivity_threshold, pd.DataFrame([temp, max(gradient_formulation_norm), max(gradient_pc_norm), max(gradient_li_norm)]).T])

    print(conductivity_threshold)

fidelity.columns = ["temperature", "gradient_formulation_norm", "gradient_pc_norm", "gradient_li_norm"]
conductivity_threshold.columns = [["temperature", "max_gradient_formulation_norm", "max_gradient_pc_norm", "max_gradient_li_norm"]]

save_dataset(fidelity, generator="fomulation_error", data_type ="all", separate=False, name=True)
save_dataset(conductivity_threshold, generator="fomulation_error", data_type ="max_gradient", separate=False, name=True)