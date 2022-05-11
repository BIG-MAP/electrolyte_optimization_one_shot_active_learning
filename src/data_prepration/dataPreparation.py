import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pandas as pd
import numpy as np
from glob import glob # https://docs.python.org/3/library/glob.html
from tqdm import tqdm # https://tqdm.github.io/
import json

SAVE = False

# the path for converting all json files to a dataframe 
dataFolder = os.path.join("data", "formulation_suggestions", "Conductivity_KIT_prediction01")

# Initialise the dataframe
raw_data = pd.DataFrame(columns=["experimentID", "electrolyteLabel", "PC", "EC", "EMC", "LiPF6", "inverseTemperature", "temperature", "conductivity", "resistance", "Z'", "Z''", "frequency", "electrolyteAmount"])

# Go through all files in the folder dataFolder
for filepath in tqdm(glob(dataFolder+"/*.json")):
    with open(filepath, "r") as fp:
        data = json.load(fp)
        # extract the data from the file
        expID = data["Experiment ID"]
        electrolyteLabel = data["Electrolyte"]["Label"]
        # initialise the composition to 0 of each component and generate a dict
        pc = 0.0
        ec = 0.0
        emc = 0.0
        lipf6 = 0.0
        composition = {"PC": pc, "EC": ec, "EMC": emc, "LiPF6": lipf6}
        # catch the difference in spelling of the keys between files
        if "Electrolyte component" in data["Electrolyte"].keys():
            electrolyteComponents = data["Electrolyte"]["Electrolyte component"]
        else:
            electrolyteComponents = data["Electrolyte"]["Electrolyte Component"]
        # go through the components and update the composition dictionary
        for component in electrolyteComponents:
                acronym = component["Acronym"]
                composition[acronym] = component["Amount"]
        # get the data from the conductivity measurements
        inverseTemperature = data["Conductivity experiment"]["Conductivity data"]["Inverse temperature"]["Data"]
        conductivity = data["Conductivity experiment"]["Conductivity data"]["Conductivity"]["Data"]
        electrolyteAmount = data["Conductivity experiment"]["Electrolyte amount"]
        # get the data from the Arrhenius plot
        temperature = data["Conductivity experiment"]["Arrhenius plot"]["Temperature"]["Data"]
        resistance = data["Conductivity experiment"]["Arrhenius plot"]["Resistance"]["Data"]

        # get the data from the impedance measurements
        Zprime = []
        ZdoublePrime = []
        frequency = []

        for measurement in data["Conductivity experiment"]["Impedance data"]:
            Zprime.append(measurement["Z'"]["Data"])
            ZdoublePrime.append(measurement["Z''"]["Data"])
            frequency.append(measurement["Frequency"]["Data"])
        # expand the data, where necessary
        length = len(temperature)
        experimentID = np.full(length, expID)    # https://numpy.org/doc/stable/reference/generated/numpy.full.html
        electrolyteLabel = np.full(length, electrolyteLabel)
        pc = np.full(length, composition["PC"])
        ec = np.full(length, composition["EC"])
        emc = np.full(length, composition["EMC"])
        lipf6 = np.full(length, composition["LiPF6"])
        electrolyteAmount = np.full(length, electrolyteAmount)

        # assemble the dataframe for this file
        thisMeasurement = pd.DataFrame(data={"experimentID": experimentID, "electrolyteLabel": electrolyteLabel, "PC": pc, "EC": ec, "EMC": emc, "LiPF6": lipf6, "inverseTemperature": inverseTemperature, "temperature": temperature, "conductivity": conductivity, "resistance": resistance, "Z'": Zprime, "Z''": ZdoublePrime, "frequency":frequency, "electrolyteAmount": electrolyteAmount})
        # add the new entry to the raw_data dataframe
        raw_data = pd.concat([raw_data, thisMeasurement], ignore_index = True, axis=0)

print(raw_data)
if SAVE:
    raw_data.to_csv(os.path.join("data", "original", "ElectrolytepredictionAfterOneShotRaw.csv"), sep=";")