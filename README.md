# Hackathon

First, I used Pytorch with the framework MONAI to do all the data collection, processing, and modeling.

So, I started with the idea of training a 3DUNET but I got several bad results.
So, in my second model tentative, I went to the SwinUNETR. This one performed the best, however still not as good as the others.

I was not able to think of an idea to use the lung segmentations to help somehow the model. 

I tried to create a model concatenating the output of different models in the following schema:

![COMPETITION IPMED](https://github.com/mathparracho/PSCC_datachallenge/assets/58774388/d87ad654-631a-4a5a-9657-22fb34e9e356)

I trained 3 different models to create this architecture and the creation class is in the notebook "experiment.ipynb"
However turns out the model was very sensible to seed variations and it performed worse in comparison to the vanilla SwinUNETR. Frankly, I do not know the reason for this behavior but it is what it is.

So, to the final Kaggle score, the SwinUNETR model with 10 epochs was the best. And this is the model I uploaded in this repository

## Files:

I organized the script that gave me the final score into a Jupyter Notebook in the folder "notebooks". There you are going to find the organized scripts.

In the folder "scripts" I uploaded all the files I used to test and create all my experiments - It is a mess! But experiments are just like that!

## Important:

All the data is in "notebooks".

If you want to re-run the model, please make sure all paths are right to get the data and the models.
I did not upload the data into this repository.
