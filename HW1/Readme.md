Data loading:
First, flatten the labels data into one dimension. And flatten the features data into 40 dimensions. And for the dataloader, the get_item part gets the previous 12 and later 12 frames together with i frame to output the final data for i.

Data preprocessing:
First, for each frame, the model uses PCA to reduce the 40 dimensions to 20 dimensions. And then the model take the previous 12 frames and the later 12 frames together as the features for the training data. 

Model trianing:
The model have 7 layers(including output layer) and 5 epochs. The input dimension is 20*(12+12+1) = 500, the output dimension is 138. Layer 1, 3, 5: Linear + BatchNorm + Dropout(p = 0.1) + PRelu; Layer 2, 4, 6: Linear + BatchNorm + Dropout(p = 0.2) + PRelu. The hidden layers' dimensions are: 2048, 1800, 1448, 1024, 724, 424, 138. 

Instructions:
1. Run training.py file.
2. Run predicting.py file.
3. Put the 1_pca.csv file's value into the submision csv.


