## Deep Learning lab tutorial (Based on Pytorch framework)


## Practice 1 Pipeline:

Practice & progress from building Convolutional Neural Network(CNN) --> Looking for proper dataet --> Training model --> validation

## Dataset was from Kaggle: <https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset>
============================================================================================

Introduction:
In this basic classification practice, I used an ez structured CNN to extract hand-crafted features and a single 2-class FC layer to finish the classfication work based on cats vs dogs dataset. Details were listed in the py file. This code included classfication, saving the best-result pth model and plotting the accuracy plot when training epochs increase.

Note: 

I used sampler in DataLoader function, definitely this part can be utlized as DataLoader(dataset,shuffle = True, batch_size = 32).

In addition, you can select to use resnet to improve classification efficiency. Considering this network is not deep, I practiced handwriting network structure.
If you would like to use resnet from torchvision.models,that's much more convienient,such as:

model = torchvision.models.resnet18()

num_in_features = model.fc.infeatures

model.fc = nn.Linear(num_in_features,2)

model.to(device)

============================================================================================
