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
- **c2.cp.reactome.v2023.1.Hs.symbols.gmt:** Canonical Pathways gene sets derived from the PID pathway database.
- **c3.all.v2023.1.Hs.symbols.gmt:** Gene sets representing potential targets of regulation by transcription factors or microRNAs.
- **c5.go.bp.v2023.1.Hs.symbols.gmt:** Gene sets derived from the GO Biological Process ontology.
- **c7.all.v2023.1.Hs.symbols.gmt:** Gene sets that represent cell states and perturbations within the immune system.