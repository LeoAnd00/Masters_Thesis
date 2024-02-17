import os
import torch
import torch.nn as nn
import pathlib
import json
import optuna
from .functions import train as trainer_fun
from .functions import predict as predict_fun
from .functions import make_cell_type_representations as generate_representation_fun
from .models import Model1 as Model1
from .models import Model2 as Model2
from .models import Model3 as Model3
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
              num_trials: int=100,
              device: str=None,
              validation_pct: float=0.2,
              gene_set_gene_limit: int=10,
              seed: int=42,
              batch_size: int=236,
              init_lr: float=0.001,
              epochs: int=100,
              lr_scheduler_warmup: int=4,
              lr_scheduler_maxiters: int=100,
              eval_freq: int=1,
              earlystopping_threshold: int=10,
              accum_grad: int = 1,
              batch_size_classifier: int = 256,
              init_lr_classifier: float = 0.001,
              lr_scheduler_warmup_classifier: int = 4,
              lr_scheduler_maxiters_classifier: int = 50,
              eval_freq_classifier: int = 1,
              epochs_classifier: int = 50,
              earlystopping_threshold_classifier: int = 10,
              accum_grad_classifier: int = 1):
        
        if self.model_name == "Model1":

            train_env = trainer_fun.train_module(data_path=adata,
                                                 save_model_path=self.model_path,
                                                 HVG=True,
                                                 HVGs=self.HVGs,
                                                 target_key=self.target_key,
                                                 batch_keys=[self.batch_key],
                                                 validation_pct=validation_pct)
            
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
                                                 gene2vec_path=self.gene2vec_path)
            
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
                                                 gene2vec_path=self.gene2vec_path)
            
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
                    n_neurons_layer1 = trial.suggest_int('n_neurons_layer1', 128, 1024, step=128)
                    n_neurons_layer2 = trial.suggest_int('n_neurons_layer2', 128, 1024, step=128)
                    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
                    dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.1)

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
                                                        only_print_best=True)
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

            else:
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
                'first_layer_dim': 256,
                'second_layer_dim': 256,
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
                detect_unknowns: bool=True,
                unknown_threshold: float=0.5,
                return_pred_probs: bool=False):
        
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

        representations = generate_representation_fun.generate_representation(data_=adata, 
                                                                              model=model, 
                                                                              model_path=self.model_path, 
                                                                              target_key=self.target_key,
                                                                              save_path=save_path, 
                                                                              batch_size=batch_size, 
                                                                              method=method)
    
        return representations
    
    def get_gene_set(self,
                gene_set_name: str):
    
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



