## Face app

Today i am going to show you how to train your own face recognition model for free, yes we are going to use **Google Colab** free GPU support to train the model.

> **Here's the challenge**
>
> We only have single image per person

The main concept is, we will use Keras data augmentation API to generate pseudo training images, not to mention we are going to use parallel programming techniques to augment and extract embeddings faster  making full use of all the free resource available on Colab.

like any other tutorials we are not going to make any dedicated colab notebook, instead we will just git clone this repository on colab and do our stuff.