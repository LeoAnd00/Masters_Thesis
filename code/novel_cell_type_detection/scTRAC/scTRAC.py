import os
import torch
import torch.nn as nn
import numpy as np
import random
import pathlib
import json
import optuna
from .functions import train as trainer_fun
from .functions import predict as predict_fun
from .functions import make_cell_type_representations as generate_representation_fun
from .models import Model1 as Model1
from .models import Model2 as Model2
from .models import Model3 as Model3
from .models import Model4 as Model4
from .models import ModelClassifier as ModelClassifier

class scTRAC():

    def __init__(self,
                 target_key: str,
                 batch_key: str,
                 latent_dim: int=100,
                 HVGs: int=2000,
                 num_HVG_buckets: int=1000,
                 num_gene_sets: int=500,
                 model_name: str="Model3",
                 model_path: str="trained_models/",
                 gene_set_name: str="c5"):
        """
        scTRAC is a machine learning model designed to generate a lower dimensional latent space for scRNA-Seq data.\n
        It can also be used for classifying cell types and identify potentially novel cell types.

        Parameters
        ----------
        target_key
            Specify key in adata.obs that contain target labels. For example "cell type".
        batch_key
            Specify key in adata.obs that contain batch effect key one wants to correct for. For example "patientID".
        latent_dim
            Dimension of latent space produced by scTRAC. Default is 100.
        HVGs
            Number of highly variable genes (HVGs) to select as input to scTRAC. Default is 2000.
        num_HVG_buckets
            Number of buckets the expression levels will be divided into. Default is 1000.
        num_gene_sets
            Number of gene sets to use. Default is 500.
        model_name
            Name of model to use (Options: "Model1", "Model2", "Model3"). Default is "Model3".
        model_path
            Path where model will be saved. Default is "trained_models/".
        gene_set_name
            Name of gene set file to use (Options: "c2", "c3", "c5", "c7", "c8", "all"). Default is "c5".

        Latent Space Example
        --------
        >>> model = scTRAC.scTRAC(target_key="cell_type", batch_key="batch")
        >>> model.train(adata=adata_train)
        >>> predictions = model.predict(adata=adata_test)

        Classifier Example
        --------
        >>> model = scTRAC.scTRAC(target_key="cell_type", batch_key="batch")
        >>> model.train(adata=adata_train, train_classifier=True)
        >>> predictions = model.predict(adata=adata_test, use_classifier=True, detect_unknowns=True)

        Returns
        -------
        None
        """
        
        self.model_name = model_name
        self.model_path = model_path + f"{self.model_name}/"
        self.target_key = target_key
        self.batch_key = batch_key
        self.latent_dim = latent_dim
        self.HVGs = HVGs
        self.num_HVG_buckets = num_HVG_buckets
        self.num_gene_sets = num_gene_sets
        self.gene_set_path = self.get_gene_set(gene_set_name=gene_set_name)

        root = pathlib.Path(__file__).parent
        self.gene2vec_path = root / "resources/gene2vec_dim_200_iter_9_w2v.txt"

        if not os.path.exists(f'{self.model_path}config/'):
            os.makedirs(f'{self.model_path}config/')


    def train(self,
              adata,
              train_classifier: bool=False,
              optimize_classifier: bool=True,
              only_print_best: bool=False,
              num_trials: int=100,
              use_already_trained_latent_space_generator: bool=False,
              device: str=None,
              validation_pct: float=0.2,
              gene_set_gene_limit: int=10,
              seed: int=42,
              batch_size: int=256,
              init_lr: float=0.001,
              epochs: int=50,
              lr_scheduler_warmup: int=4,
              lr_scheduler_maxiters: int=50,
              eval_freq: int=1,
              earlystopping_threshold: int=20,
              accum_grad: int = 1,
              batch_size_classifier: int = 256,
              init_lr_classifier: float = 0.001,
              lr_scheduler_warmup_classifier: int = 4,
              lr_scheduler_maxiters_classifier: int = 50,
              eval_freq_classifier: int = 1,
              epochs_classifier: int = 50,
              earlystopping_threshold_classifier: int = 10,
              accum_grad_classifier: int = 1):
        """
        Fit scTRAC to your Anndata.\n
        Saves model and relevant information to be able to make predictions on new data.

        Parameters
        ----------
        adata 
            An AnnData object containing single-cell RNA-seq data.
        train_classifier
            Whether to train scTRAC as a classifier (True) or to produce a latent space (False). Default is False.
        optimize_classifier
            Whether to use Optuna to optimize the classifier part of the model, assuming train_classifier is True. Default is True.
        only_print_best
            Whether to only print the results of the best epoch of each trial (True) or print performance at each epoch (False).
            Default is False.
        num_trials
            Number of trials for optimizing classifier, assuming train_classifier and optimize_classifier are True. Default is 100.
        use_already_trained_latent_space_generator
            If you've already trained scTRAC on making a latent space you can use this model when training the classifier (True),\n 
            or if you haven't trained it you can train it as a first step of training the classifier (False). Default is False.
        device
            Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".\n
            Default is None.
        validation_pct
            The percentage of data used for validation. Default is 0.2, meaning 20%.
        gene_set_gene_limit
            Minimum number of HVGs a gene set must have to be considered. Default is 10.
        seed
            Which random seed to use. Default is 42.
        batch_size
            Mini-batch size used for training latent space producing part of scTRAC. Default is 236.
        init_lr
            Initial learning rate for training latent space producing part of scTRAC. Default is 0.001.
        epochs
            Number of epochs for training latent space producing part of scTRAC. Default is 100.
        lr_scheduler_warmup
            Number of epochs for the warm up part of the CosineWarmupScheduler for training latent space producing part of scTRAC.\n
            Default is 4.
        lr_scheduler_maxiters
            Number of epochs at which the learning rate would become zero for training latent space producing part of scTRAC.\n
            Default is 110.
        eval_freq
            Number of epochs between calculating loss of validation data for training latent space producing part of scTRAC.\n 
            Default is 1.
        earlystopping_threshold
            Number of validated epochs before terminating training if no improvements to the validation loss is made for training\n 
            latent space producing part of scTRAC. Default is 20.
        accum_grad
            Number of Mini-batches to calculate gradient for before updating weights for training latent space producing part of\n 
            scTRAC. Default is 1.
        batch_size_classifier
            Mini-batch size used for training classifier part of scTRAC. Default is 256.
        init_lr_classifier
            Initial learning rate for training classifier part of scTRAC. Default is 0.001.
        epochs_classifier
            Number of epochs for training classifier part of scTRAC. Default is 50.
        lr_scheduler_warmup_classifier
            Number of epochs for the warm up part of the CosineWarmupScheduler for training classifier part of scTRAC.\n
            Default is 4.
        lr_scheduler_maxiters_classifier
            Number of epochs at which the learning rate would become zero for training classifier part of scTRAC.\n
            Default is 50.
        eval_freq_classifier
            Number of epochs between calculating loss of validation data for training classifier part of scTRAC.\n 
            Default is 1.
        earlystopping_threshold_classifier
            Number of validated epochs before terminating training if no improvements to the validation loss is made for training\n 
            classifier part of scTRAC. Default is 10.
        accum_grad_classifier
            Number of Mini-batches to calculate gradient for before updating weights for training classifier part of\n 
            scTRAC. Default is 1.

        Returns
        -------
        None
        """

        if adata.n_vars < self.HVGs:
            raise ValueError('Number of genes in adata is less than number of HVGs specified to be used.')
        
        if self.model_name == "Model1":

            train_env = trainer_fun.train_module(data_path=adata,
                                                 save_model_path=self.model_path,
                                                 HVG=True,
                                                 HVGs=self.HVGs,
                                                 target_key=self.target_key,
                                                 batch_keys=[self.batch_key],
                                                 validation_pct=validation_pct)
            
            self.rep_seed(seed=seed)
            model = Model1.Model1(input_dim=self.HVGs,
                                  output_dim=self.latent_dim)
            
            # Sample configuration dictionary
            config = {
                'input_dim': self.HVGs,
                'output_dim': self.latent_dim
            }

            # Define the file path to save the configuration
            config_file_path = f'{self.model_path}config/model_config.json'

            # Save the configuration dictionary to a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)
            
        elif self.model_name == "Model2":

            train_env = trainer_fun.train_module(data_path=adata,
                                                 pathways_file_path=None,
                                                 num_pathways=self.num_gene_sets,
                                                 pathway_gene_limit=gene_set_gene_limit,
                                                 save_model_path=self.model_path,
                                                 HVG=True,
                                                 HVGs=self.HVGs,
                                                 HVG_buckets=self.num_HVG_buckets,
                                                 use_HVG_buckets=True,
                                                 target_key=self.target_key,
                                                 batch_keys=[self.batch_key],
                                                 use_gene2vec_emb=True,
                                                 gene2vec_path=self.gene2vec_path,
                                                 validation_pct=validation_pct)
            
            self.rep_seed(seed=seed)
            model = Model2.Model2(num_HVGs=self.HVGs,
                                  output_dim=self.latent_dim,
                                  HVG_tokens=self.num_HVG_buckets,
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)
            
            # Sample configuration dictionary
            config = {
                'input_dim': self.HVGs,
                'output_dim': self.latent_dim,
                'HVG_tokens': self.num_HVG_buckets
            }

            # Define the file path to save the configuration
            config_file_path = f'{self.model_path}config/model_config.json'

            # Save the configuration dictionary to a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)
        
        elif self.model_name == "Model3":

            train_env = trainer_fun.train_module(data_path=adata,
                                                 pathways_file_path=self.gene_set_path,
                                                 num_pathways=self.num_gene_sets,
                                                 pathway_gene_limit=gene_set_gene_limit,
                                                 save_model_path=self.model_path,
                                                 HVG=True,
                                                 HVGs=self.HVGs,
                                                 HVG_buckets=self.num_HVG_buckets,
                                                 use_HVG_buckets=True,
                                                 target_key=self.target_key,
                                                 batch_keys=[self.batch_key],
                                                 use_gene2vec_emb=True,
                                                 gene2vec_path=self.gene2vec_path,
                                                 validation_pct=validation_pct)
            
            self.rep_seed(seed=seed)
            model = Model3.Model3(mask=train_env.data_env.pathway_mask,
                                  num_HVGs=self.HVGs,
                                  output_dim=self.latent_dim,
                                  num_pathways=self.num_gene_sets,
                                  HVG_tokens=self.num_HVG_buckets,
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)
            
            # Sample configuration dictionary
            config = {
                'input_dim': self.HVGs,
                'output_dim': self.latent_dim,
                'HVG_tokens': self.num_HVG_buckets,
                'num_pathways': self.num_gene_sets
            }

            # Define the file path to save the configuration
            config_file_path = f'{self.model_path}config/model_config.json'

            # Save the configuration dictionary to a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)

        elif self.model_name == "Model4":

            train_env = trainer_fun.train_module(data_path=adata,
                                                 pathways_file_path=self.gene_set_path,
                                                 num_pathways=self.num_gene_sets,
                                                 pathway_gene_limit=gene_set_gene_limit,
                                                 save_model_path=self.model_path,
                                                 HVG=True,
                                                 HVGs=self.HVGs,
                                                 HVG_buckets=self.num_HVG_buckets,
                                                 use_HVG_buckets=True,
                                                 target_key=self.target_key,
                                                 batch_keys=[self.batch_key],
                                                 use_gene2vec_emb=True,
                                                 gene2vec_path=self.gene2vec_path,
                                                 validation_pct=validation_pct)
            
            self.rep_seed(seed=seed)
            model = Model4.Model4(mask=train_env.data_env.pathway_mask,
                                  num_HVGs=self.HVGs,
                                  output_dim=self.latent_dim,
                                  num_pathways=self.num_gene_sets,
                                  HVG_tokens=self.num_HVG_buckets,
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)
            
            # Sample configuration dictionary
            config = {
                'input_dim': self.HVGs,
                'output_dim': self.latent_dim,
                'HVG_tokens': self.num_HVG_buckets,
                'num_pathways': self.num_gene_sets
            }

            # Define the file path to save the configuration
            config_file_path = f'{self.model_path}config/model_config.json'

            # Save the configuration dictionary to a JSON file
            with open(config_file_path, 'w') as f:
                json.dump(config, f, indent=4)

        if use_already_trained_latent_space_generator == False: 
            train_env.train(model=model,
                            model_name=self.model_name,
                            device=device,
                            seed=seed,
                            batch_size=batch_size,
                            use_target_weights=True,
                            use_batch_weights=True,
                            init_temperature=0.25,
                            min_temperature=0.1,
                            max_temperature=2.0,
                            init_lr=init_lr,
                            lr_scheduler_warmup=lr_scheduler_warmup,
                            lr_scheduler_maxiters=lr_scheduler_maxiters,
                            eval_freq=eval_freq,
                            epochs=epochs,
                            earlystopping_threshold=earlystopping_threshold,
                            accum_grad=accum_grad)
                        
        
        if train_classifier:
            if optimize_classifier:
                def objective(trial):

                    # Parameters to optimize
                    n_neurons_layer1 = trial.suggest_int('n_neurons_layer1', 64, 2048, step=64)
                    n_neurons_layer2 = trial.suggest_int('n_neurons_layer2', 64, 2048, step=64)
                    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
                    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)

                    self.rep_seed(seed=seed)
                    model_classifier = ModelClassifier.ModelClassifier(input_dim=self.latent_dim,
                                                                       first_layer_dim=n_neurons_layer1,
                                                                       second_layer_dim=n_neurons_layer2,
                                                                       classifier_drop_out=dropout,
                                                                       num_cell_types=len(adata.obs[self.target_key].unique()))
                
                    val_loss = train_env.train_classifier(model=model,
                                                        model_name=self.model_name,
                                                        model_classifier=model_classifier,
                                                        device=device,
                                                        seed=seed,
                                                        init_lr=learning_rate,
                                                        batch_size=batch_size_classifier,
                                                        lr_scheduler_warmup=lr_scheduler_warmup_classifier,
                                                        lr_scheduler_maxiters=lr_scheduler_maxiters_classifier,
                                                        eval_freq=eval_freq_classifier,
                                                        epochs=epochs_classifier,
                                                        earlystopping_threshold=earlystopping_threshold_classifier,
                                                        accum_grad=accum_grad_classifier,
                                                        only_print_best=only_print_best)
                    return val_loss
                
                # Define the study and optimize
                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=num_trials)

                print('Number of finished trials: ', len(study.trials))
                print('Best trial:')
                trial = study.best_trial

                print('  Value: ', trial.value)
                print('  Params: ')
                opt_dict = {}
                for key, value in trial.params.items():
                    print('    {}: {}'.format(key, value))
                    opt_dict[key] = value

                # Sample configuration dictionary
                config = {
                    'input_dim': self.latent_dim,
                    'num_cell_types': len(adata.obs[self.target_key].unique()),
                    'first_layer_dim': opt_dict['n_neurons_layer1'],
                    'second_layer_dim': opt_dict['n_neurons_layer2'],
                    'classifier_drop_out': opt_dict['dropout']
                }

                # Define the file path to save the configuration
                config_file_path = f'{self.model_path}config/model_classifier_config.json'

                # Save the configuration dictionary to a JSON file
                with open(config_file_path, 'w') as f:
                    json.dump(config, f, indent=4)

                self.rep_seed(seed=seed)
                model_classifier = ModelClassifier.ModelClassifier(input_dim=config["input_dim"],
                                                               num_cell_types=config["num_cell_types"],
                                                               first_layer_dim=config["first_layer_dim"],
                                                               second_layer_dim=config["second_layer_dim"],
                                                               classifier_drop_out=config["classifier_drop_out"])
            
                _ = train_env.train_classifier(model=model,
                                                    model_name=self.model_name,
                                                    model_classifier=model_classifier,
                                                    device=device,
                                                    seed=seed,
                                                    init_lr=opt_dict["learning_rate"],
                                                    batch_size=batch_size_classifier,
                                                    lr_scheduler_warmup=lr_scheduler_warmup_classifier,
                                                    lr_scheduler_maxiters=lr_scheduler_maxiters_classifier,
                                                    eval_freq=eval_freq_classifier,
                                                    epochs=epochs_classifier,
                                                    earlystopping_threshold=earlystopping_threshold_classifier,
                                                    accum_grad=accum_grad_classifier,
                                                    only_print_best=True)

            else:
                self.rep_seed(seed=seed)
                model_classifier = ModelClassifier.ModelClassifier(input_dim=self.latent_dim,
                                                                num_cell_types=len(adata.obs[self.target_key].unique()))
                
                _ = train_env.train_classifier(model=model,
                                            model_name=self.model_name,
                                            model_classifier=model_classifier,
                                            device=device,
                                            seed=seed,
                                            init_lr=init_lr_classifier,
                                            batch_size=batch_size_classifier,
                                            lr_scheduler_warmup=lr_scheduler_warmup_classifier,
                                            lr_scheduler_maxiters=lr_scheduler_maxiters_classifier,
                                            eval_freq=eval_freq_classifier,
                                            epochs=epochs_classifier,
                                            earlystopping_threshold=earlystopping_threshold_classifier,
                                            accum_grad=accum_grad_classifier)
                
                # Sample configuration dictionary
                config = {
                    'input_dim': self.latent_dim,
                    'num_cell_types': len(adata.obs[self.target_key].unique()),
                    'first_layer_dim': 512,
                    'second_layer_dim': 512,
                    'classifier_drop_out': 0.2
                }

                # Define the file path to save the configuration
                config_file_path = f'{self.model_path}config/model_classifier_config.json'

                # Save the configuration dictionary to a JSON file
                with open(config_file_path, 'w') as f:
                    json.dump(config, f, indent=4)
                

    def predict(self,
                adata,
                batch_size: int=32, 
                device: str=None, 
                use_classifier: bool=False,
                detect_unknowns: bool=False,
                unknown_threshold: float=0.5,
                return_pred_probs: bool=False):
        """
        Make predictions using scTRAC.\n
        Make sure you've got a trained model before calling this function.

        Parameters
        ----------
        adata 
            An AnnData object containing single-cell RNA-seq data.
        batch_size
            Mini-batch size used for making predictions. Default is 32.
        device
            Which device to use, like "cpu" or "cuda". If left as None it will automatically select "cuda" if available, else "cpu".\n
            Default is None.
        use_classifier
            Whether to make cell type prediction using classifier part od scTRAC (True) or predict latent space (False). Default is False.
        detect_unknowns
            Whether to consider samples with a confidence below unknown_threshold as unknown/novel. Default is False.
        unknown_threshold
            Confidence threshold of which if a sample has a confidence below it, it is considered unknown/novel. Default is 0.5.
        return_pred_probs
            Whether to return the probability/confidence of scTRAC cell type predictions. Default is False.

        Returns
        -------
        If return_pred_probs == False:
            return pred
        If return_pred_probs == True:
            return pred, pred_prob
        """
        
        # Define the file path from which to load the configuration
        config_file_path = f'{self.model_path}config/model_config.json'

        # Load the configuration from the JSON file
        with open(config_file_path, 'r') as f:
            loaded_config = json.load(f)
        
        if self.model_name == "Model1":

            model = Model1.Model1(input_dim=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"])

        elif self.model_name == "Model2":

            model = Model2.Model2(num_HVGs=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"],
                                  HVG_tokens=loaded_config["HVG_tokens"],
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)
            
        elif self.model_name == "Model3":

            if os.path.exists(f"{self.model_path}ModelMetadata/gene_set_mask.pt"):
                pathway_mask = torch.load(f"{self.model_path}ModelMetadata/gene_set_mask.pt")

            model = Model3.Model3(mask=pathway_mask,
                                  num_HVGs=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"],
                                  num_pathways=loaded_config["num_pathways"],
                                  HVG_tokens=loaded_config["HVG_tokens"],
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)

        elif self.model_name == "Model4":

            if os.path.exists(f"{self.model_path}ModelMetadata/gene_set_mask.pt"):
                pathway_mask = torch.load(f"{self.model_path}ModelMetadata/gene_set_mask.pt")

            model = Model4.Model4(mask=pathway_mask,
                                  num_HVGs=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"],
                                  num_pathways=loaded_config["num_pathways"],
                                  HVG_tokens=loaded_config["HVG_tokens"],
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)

        if use_classifier:

            # Define the file path from which to load the configuration
            config_file_path = f'{self.model_path}config/model_classifier_config.json'

            # Load the configuration from the JSON file
            with open(config_file_path, 'r') as f:
                loaded_config_classifier = json.load(f)
            
            model_classifier = ModelClassifier.ModelClassifier(input_dim=loaded_config_classifier["input_dim"],
                                                               num_cell_types=loaded_config_classifier["num_cell_types"],
                                                               first_layer_dim=loaded_config_classifier["first_layer_dim"],
                                                               second_layer_dim=loaded_config_classifier["second_layer_dim"],
                                                               classifier_drop_out=loaded_config_classifier["classifier_drop_out"])
            
            pred, pred_prob = predict_fun.predict(data_=adata,
                                                  model_name=self.model_name,
                                                  model_path=self.model_path,
                                                  model=model,
                                                  model_classifier=model_classifier,
                                                  batch_size=batch_size,
                                                  device=device,
                                                  use_classifier=use_classifier,
                                                  detect_unknowns=detect_unknowns,
                                                  unknown_threshold=unknown_threshold)
        else:
            model_classifier = None

            pred = predict_fun.predict(data_=adata,
                                       model_name=self.model_name,
                                       model_path=self.model_path,
                                       model=model,
                                       model_classifier=model_classifier,
                                       batch_size=batch_size,
                                       device=device,
                                       use_classifier=use_classifier,
                                       detect_unknowns=detect_unknowns,
                                       unknown_threshold=unknown_threshold)
        
        if return_pred_probs:
            return pred, pred_prob
        else:
            return pred

    def generate_representations(self,
                                 adata, 
                                 save_path: str="cell_type_vector_representation/CellTypeRepresentations.csv",
                                 batch_size: int=32,
                                 method: str="centroid"):
        """
        Generates cell type representation vectors using the latent space produced by scTRAC.

        Parameters
        ----------
        adata 
            An AnnData object containing single-cell RNA-seq data.
        save_path
            Path where a .csv file containing the vector representation of each cell type will be saved.\n
            Default is "cell_type_vector_representation/CellTypeRepresentations.csv"
        batch_size
            Mini-batch size used for making predictions. Default is 32.
        method
            Which method to use for making representations (Options: "centroid", "median", "medoid"). Default is "centroid".

        Returns
        -------
        representations
        """
        
        # Define the file path from which to load the configuration
        config_file_path = f'{self.model_path}config/model_config.json'

        # Load the configuration from the JSON file
        with open(config_file_path, 'r') as f:
            loaded_config = json.load(f)
        
        if self.model_name == "Model1":

            model = Model1.Model1(input_dim=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"])

        elif self.model_name == "Model2":

            model = Model2.Model2(num_HVGs=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"],
                                  HVG_tokens=loaded_config["HVG_tokens"],
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)
            
        elif self.model_name == "Model3":

            if os.path.exists(f"{self.model_path}ModelMetadata/gene_set_mask.pt"):
                pathway_mask = torch.load(f"{self.model_path}ModelMetadata/gene_set_mask.pt")

            model = Model3.Model3(mask=pathway_mask,
                                  num_HVGs=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"],
                                  num_pathways=loaded_config["num_pathways"],
                                  HVG_tokens=loaded_config["HVG_tokens"],
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)

        elif self.model_name == "Model4":

            if os.path.exists(f"{self.model_path}ModelMetadata/gene_set_mask.pt"):
                pathway_mask = torch.load(f"{self.model_path}ModelMetadata/gene_set_mask.pt")

            model = Model4.Model4(mask=pathway_mask,
                                  num_HVGs=loaded_config["input_dim"],
                                  output_dim=loaded_config["output_dim"],
                                  num_pathways=loaded_config["num_pathways"],
                                  HVG_tokens=loaded_config["HVG_tokens"],
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)

        representations = generate_representation_fun.generate_representation(data_=adata, 
                                                                              model=model, 
                                                                              model_name=self.model_name,
                                                                              model_path=self.model_path, 
                                                                              target_key=self.target_key,
                                                                              save_path=save_path, 
                                                                              batch_size=batch_size, 
                                                                              method=method)
    
        return representations
    
    def get_gene_set(self, gene_set_name: str):
        """
        Makes path to specified gene set dataset.

        Parameters
        ----------
        gene_set_name
            Specify whether to use "c2", "c3", "c5", "c7", "c8", or "all" gene set.

        Returns
        -------
        gene_set_file_name
        """
    
        root = pathlib.Path(__file__).parent
        gene_set_files = {
            "all": [root / "resources/all_pathways.json"],
            "c2": [root / "resources/c2_pathways.json"],
            "c3": [root / "resources/c3_pathways.json"],
            "c5": [root / "resources/c5_pathways.json"],
            "c7": [root / "resources/c7_pathways.json"],
            "c8": [root / "resources/c8_pathways.json"]
        }
        return gene_set_files[gene_set_name][0]
    
    def rep_seed(self, seed=42):
        """
        Sets the random seed for torch, random and numpy.

        Parameters
        ----------
        seed
            Which random seed to use. Default is 42.

        Returns
        -------
        None
        """
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



