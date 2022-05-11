from utils import *
class Database:
    def __init__(self):
        """importing two datasets before and ater one shot active learning 
            and get their corresponding settings
        """
        # dataset before one shot active learning 
        self._initial_data = load_dataset(dataset_type="original", name="OriginalData.csv")
        # dataset after one shot active learning 
        self._updated_data = load_dataset(dataset_type="original", name="OriginalData2.csv")
        # the added electrolyte formulations
        self._added_formulation = load_dataset(dataset_type="formulation_suggestions", name="comparison_experiment_finale.csv")
        # extracting the temperatures and sort them 
        self.temperatures = self._updated_data["temperature"].unique().tolist()
        self.temperatures.sort()
        self.sorted_temperatures = self.temperatures
        # specify the input & output columns used for training
        self.input_column = ["PcEcPc", "liEcPc"]
        self.output_column = ["conductivity"]
    
    def __str__(self):
        return f"Variaty of electrolyte formulations for LIBs"

