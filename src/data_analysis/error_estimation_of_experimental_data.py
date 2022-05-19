''' This file loads the experimental data and determines the maximum range for 5 repetitions of an experiment. '''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pandas as pd
import numpy as np
from database import Database

## Load the data
data_config = Database()
data = data_config._updated_data
data.set_index('experimentID', inplace=True)
idUniqueSeries = pd.Series([eID.split('_')[-2] for eID in data.index]).unique()

# Go through all unique electrolyte labels
for id in idUniqueSeries:
    # Get all repetition for this ID
    indices = pd.Series([ind for ind in data.index if id in ind]).unique()
    # Get the unique temperatures available for these IDS
    temps = data.loc[indices, 'temperature'].unique()
    # Prepare the dataframe to store single experiment data
    data_singleExperiment = pd.DataFrame(data.loc[indices, ['conductivity', 'temperature']])
    # Set the index to the temperature
    data_singleExperiment.reset_index(inplace=True)
    data_singleExperiment.set_index('temperature',inplace=True)

    # Go through all the temperatures
    for t in temps:
        # Filter the data for a single experiment for a single temperature
        data_singleExperiment_singleT = data_singleExperiment.loc[t,:]
        try:
            # Try to get the min and max value
            minValue_singleExperiment = data_singleExperiment_singleT.loc[:, 'conductivity'].min()
            maxValue_singleExperiment = data_singleExperiment_singleT.loc[:, 'conductivity'].max()
        except:
            # If this does not work, there is only one value and a Series is returned
            data_singleExperiment_singleT = pd.DataFrame(data_singleExperiment_singleT).transpose()
            # If only one measurement is available, set the value to NaN to enable min and max afterwards
            minValue_singleExperiment = np.NaN 
            maxValue_singleExperiment = np.NaN
        # Copy the dataframe for single experiment and single time to get rid of slicing error
        data_singleExperiment_singleT = data_singleExperiment_singleT.copy()
        # Put the range into the dataframe for single experiment and single temperature
        data_singleExperiment_singleT.loc[:, 'range'] = np.full((data_singleExperiment_singleT.shape[0],), maxValue_singleExperiment - minValue_singleExperiment)   # This line is buggy!
        # Set the index to experimentID
        data_singleExperiment_singleT.reset_index(inplace=True)
        data_singleExperiment_singleT.set_index('experimentID', inplace=True)
        #print(data_singleExperiment_singleT)
        # Put the data in the data dataframe to collect them all
        data.loc[[(id in i) for i in data.index] & (data['temperature']==t), 'range'] = data_singleExperiment_singleT['range']

# Prepare a dataframe for collecting all the max and min ranges
ranges = pd.DataFrame(index=list(data['temperature'].unique()) + ['total'], columns=['minRange', 'maxRange', 'medianRange'])
# Fill the ranges dataframe going through all temperatures
for t in data['temperature'].unique():
    # Save min and max for each temperature to dataframe
    ranges.loc[t, 'maxRange'] = data.loc[data['temperature']==t, 'range'].max()
    ranges.loc[t, 'minRange'] = data.loc[data['temperature']==t, 'range'].min()
    ranges.loc[t, 'medianRange'] = data.loc[data['temperature']==t, 'range'].median()

# Save the min and max of total range data to the ranges dataframe
ranges.loc['total', 'minRange'] = ranges.loc[[i for i in ranges.index if i != 'total'], 'minRange'].min()
ranges.loc['total', 'maxRange'] = ranges.loc[[i for i in ranges.index if i != 'total'], 'maxRange'].max()
ranges.loc['total', 'medianRange'] = data.loc[:, 'range'].median()

print(ranges)
# Save the dataframe as csv
ranges.to_csv(os.path.join("data\augmented\regression\combined_datasets\fomulation_error\expertimental_error", "ranges.csv"), sep=';')