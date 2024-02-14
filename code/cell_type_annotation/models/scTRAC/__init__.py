import os
import torch
import torch.nn as nn
import pathlib
from functions import train as trainer_fun
from functions.predict import predict as predict_fun
from functions.make_cell_type_representations import generate_representation as generate_representation_fun
from models import Model1 as Model1
from models import Model2 as Model2
from models import Model3 as Model3
from models import ModelClassifier as ModelClassifier

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


    def train(self,
              adata,
              train_classifier: bool=False,
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
              lr_scheduler_maxiters_classifier: int = 100,
              eval_freq_classifier: int = 1,
              epochs_classifier: int = 100,
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
            
            model = Model1.Model1(output_dim=self.latent_dim)
            
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
            
        train_env.train(model=model,
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
                        accum_grad=accum_grad,
                        train_classifier=train_classifier,
                        batch_size_classifier=batch_size_classifier,
                        init_lr_classifier=init_lr_classifier,
                        lr_scheduler_warmup_classifier=lr_scheduler_warmup_classifier,
                        lr_scheduler_maxiters_classifier=lr_scheduler_maxiters_classifier,
                        eval_freq_classifier=eval_freq_classifier,
                        epochs_classifier=epochs_classifier,
                        earlystopping_threshold_classifier=earlystopping_threshold_classifier,
                        accum_grad_classifier=accum_grad_classifier)

    def predict(self,
                adata,
                batch_size: int=32, 
                device: str=None, 
                use_classifier: bool=False,
                detect_unknowns: bool=True,
                unknown_threshold: float=0.5,
                return_pred_probs: bool=False):
        
        if self.model_name == "Model1":

            model = Model1.Model1(output_dim=self.latent_dim)

        elif self.model_name == "Model2":

            model = Model2.Model2(num_HVGs=self.HVGs,
                                  output_dim=self.latent_dim,
                                  HVG_tokens=self.num_HVG_buckets,
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)
            
        elif self.model_name == "Model3":

            if os.path.exists(f"{self.model_path}/ModelMetadata/gene_set_mask.pt"):
                pathway_mask = torch.load(f"{self.model_path}/ModelMetadata/gene_set_mask.pt")

            model = Model3.Model3(mask=pathway_mask,
                                  num_HVGs=self.HVGs,
                                  output_dim=self.latent_dim,
                                  num_pathways=self.num_gene_sets,
                                  HVG_tokens=self.num_HVG_buckets,
                                  HVG_embedding_dim=200,
                                  use_gene2vec_emb=True)

        model_classifier = ModelClassifier.ModelClassifier(input_dim=self.latent_dim,
                                                           num_cell_types=len(adata.obs[self.target_key].unique()))
        
        pred, pred_prob = predict_fun(data_=adata,
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

        representations = generate_representation_fun(data_=adata, 
                                                    model=self.model, 
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



