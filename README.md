## Deep Learning lab tutorial (Based on Pytorch framework)
## Practice 1 Pipeline:

Practice & progress from building Convolutional Neural Network(CNN) --> Looking for proper dataet --> Training model --> validation

## Dataset was from Kaggle: <https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset>
====================================================================================================

Introduction:
In this basic classification practice, I used an ez structured CNN to extract hand-crafted features and a single 2-class FC layer to finish the classfication work based on cats vs dogs dataset. Details were listed in the py file. This code included classfication, saving the best-result pth model and plotting the accuracy plot when training epochs increase.

Note: 
I used sampler in DataLoader function, definitely this part can be utlized as DataLoader(dataset,shuffle = True, batch_size = 32).
