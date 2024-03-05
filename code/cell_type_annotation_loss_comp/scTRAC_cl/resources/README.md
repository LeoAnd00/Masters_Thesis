# gene2vec_embeddings
Contains 200 dimensional gene embeddings for human genes.
<br>
Made by [Zou. Q. et al.](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-018-5370-x).
<br>
Code to run [Gene2vec](https://github.com/jingcheng-du/Gene2vec/tree/master).

# pathway_infromation
Contains information on which human genes are realted in different pathways in human cells.
<br>
Pathway information can be downloaded from [GSEA](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp).

## Structur
All gene set files below have been filtered so that only gene sets with less than 300 genes are kept.
- **c2_pathways.json:** Canonical Pathways gene sets derived from the PID pathway database.
- **c3_pathways.json:** Gene sets representing potential targets of regulation by transcription factors or microRNAs.
- **c5_pathways.json:** Gene sets derived from the GO Biological Process ontology.
- **c7_pathways.json:** Gene sets that represent cell states and perturbations within the immune system.
- **c8_pathways.json:** Gene sets that contain curated cluster markers for cell types identified in single-cell sequencing studies of human tissue.
- **all_pathways.json:** All gene sets above in one file.