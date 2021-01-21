## This is about the sampling in Baysien network for the final project of Probability Graphical Model Course
The python scripts is for building the probability model(Baysien network),  sampling and the compute the MLE. The report is also included.
To automatic generate the probability models in one batch, you can use builBN in bn_builder.py:
    ```
    builBN(hand_addG,number_of_edge_list,skew,merge_component=True)
    ```  
```hand_addG``` is a dictionary with the model we built by our hand already.   
```number_of_edge_list``` is the edge we want to add to the probability models which is generated automatically.   
```skew```  is making the distribution skwe or not, with geometric sequence of five for alpha series.   
```merge_component``` wrapped the probability graphical model to the dictionary as a output or not, default is true. 
