# novel_cell_type_detection
Contains all code used in this project for evaluating novel cell type prediction ability.

## How to run
In *benchmark_annotation.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train model1 for each fold when leaving out every type of cell in the MacParland dataset. <br><br>
In *benchmark_annotation_Segerstolpe.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train model1 for each fold when leaving out every type of cell in the Segerstolpe dataset. <br><br>
In *benchmark_annotation_Baron.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train model1 for each fold when leaving out every type of cell in the Baron dataset. <br><br>
In *benchmark_annotation_Zheng68k.py* the commands needed to run the novel cell type detection can be found as a comment. <br> 
In this code we train model1 for each fold when leaving out every type of cell in the Zheng68k dataset. <br><br>
In *visualization/* there's code to create visualization of min confidence for novel and non-novel cells of each fold for each dataset and each cell type dropout event. 