import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

class metrics():

    def __init__(self):

        self.metrics = None

    def calc_metrics(self, path: str, method_name: str, dataset_name: str):

        results = pd.read_csv(f'{path}.csv', index_col=0)

        unique_folds = np.unique(results.fold)

        # Define cell types to exclude
        exclude_cell_types = ['Mature_B_Cells', 
                            'Plasma_Cells', 
                            'alpha-beta_T_Cells', 
                            'gamma-delta_T_Cells_1', 
                            'gamma-delta_T_Cells_2']

        for fold_idx in unique_folds:

            results_temp = results.loc[results['fold'] == fold_idx, :]

            results_novel = results_temp[results_temp['true_label'].isin(exclude_cell_types)]
            results_novel.true_label = ["Novel"]*len(results_novel.true_label)

            results_not_novel = results_temp[~results_temp['true_label'].isin(exclude_cell_types)]

            # Extract the unique labels
            unique_labels1 = np.unique(results_temp.true_label)
            unique_labels2 = np.unique(results_temp.pred)
            unique_labels3 = np.unique(results_novel.pred)
            unique_labels = np.unique(np.concatenate([unique_labels1,unique_labels2,unique_labels3]))

            # Convert string labels to numerical labels
            label_encoder_temp = LabelEncoder()
            label_encoder_temp.fit(unique_labels)


            y_true = label_encoder_temp.transform(results_not_novel.true_label)
            y_pred = label_encoder_temp.transform(results_not_novel.pred)

            # Calculate accuracy
            #print(f"Method {method_name} | Fold {fold_idx}")
            accuracy_not_novel = accuracy_score(y_true, y_pred)
            #print("Accuracy:", accuracy)

            # Calculate balanced accuracy
            balanced_accuracy_not_novel = balanced_accuracy_score(y_true, y_pred)
            #print("Balanced Accuracy:", balanced_accuracy)

            # Calculate F1 score
            f1_not_novel = f1_score(y_true, y_pred, average='weighted')
            #print("F1 Score:", f1)

            y_true = label_encoder_temp.transform(results_novel.true_label)
            y_pred = label_encoder_temp.transform(results_novel.pred)

            # Calculate accuracy
            #print(f"Method {method_name} | Fold {fold_idx}")
            accuracy_novel = accuracy_score(y_true, y_pred)
            #print("Accuracy:", accuracy)

            # Calculate balanced accuracy
            balanced_accuracy_novel = balanced_accuracy_score(y_true, y_pred)
            #print("Balanced Accuracy:", balanced_accuracy)

            # Calculate F1 score
            f1_novel = f1_score(y_true, y_pred, average='weighted')
            #print("F1 Score:", f1)
            
            if self.metrics is None:
                self.metrics = pd.DataFrame({"method": method_name, 
                                            "accuracy": accuracy_not_novel, 
                                            "balanced_accuracy": balanced_accuracy_not_novel,
                                            "f1_score": f1_not_novel,
                                            "dataset": dataset_name,
                                            "novel": "Known",
                                            "fold": fold_idx}, index=[0])
                temp = pd.DataFrame({"method": method_name, 
                                    "accuracy": accuracy_novel, 
                                    "balanced_accuracy": balanced_accuracy_novel,
                                    "f1_score": f1_novel,
                                    "dataset": dataset_name,
                                    "novel": "Novel",
                                    "fold": fold_idx}, index=[0])
                self.metrics = pd.concat([self.metrics, temp], ignore_index=True)
            else:
                temp = pd.DataFrame({"method": method_name, 
                                    "accuracy": accuracy_not_novel, 
                                    "balanced_accuracy": balanced_accuracy_not_novel,
                                    "f1_score": f1_not_novel,
                                    "dataset": dataset_name,
                                    "novel": "Known",
                                    "fold": fold_idx}, index=[0])
                self.metrics = pd.concat([self.metrics, temp], ignore_index=True)
                temp = pd.DataFrame({"method": method_name, 
                                    "accuracy": accuracy_novel, 
                                    "balanced_accuracy": balanced_accuracy_novel,
                                    "f1_score": f1_novel,
                                    "dataset": dataset_name,
                                    "novel": "Novel",
                                    "fold": fold_idx}, index=[0])
                self.metrics = pd.concat([self.metrics, temp], ignore_index=True)

    def read_results(self, path: str):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        path : str, optional
            The file path and name of the CSV file to read.

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """
        if self.metrics is not None:
            metrics = pd.read_csv(f'{path}.csv', index_col=0)
            self.metrics = pd.concat([self.metrics, metrics], axis="rows")
        else:
            self.metrics = pd.read_csv(f'{path}.csv', index_col=0)

    def download_results(self,path: str):
        """
        Saves the performance metrics dataframe as a CSV file.

        Parameters
        ----------
        name : str, optional
            The file path and name for the CSV file (default is 'benchmarks/results/Benchmark_results' (file name will then be Benchmark_results.csv)).

        Returns
        -------
        None

        Notes
        -----
        This method exports the performance metrics dataframe to a CSV file.
        """
        self.metrics.to_csv(f'{path}.csv', index=True, header=True)
        self.metrics = None