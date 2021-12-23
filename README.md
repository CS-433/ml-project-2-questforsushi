# ml-project-2-questforsushi
ml-project-2-questforsushi created by GitHub Classroom

## Authors 
Sebastian Alveteg\
Stefan Rotarus\
Christos Fragkos

## 1. Intro 
Dear assistants, dear professors, first of all, good morning/afternoon/night!
Let us explain a bit how the following files are structured. 

## 2. Large data download
As a first step you will need to additionally download (part of) the large data files we are using. These cannot be uploaded to github, so we provide all of them in the following [drive](https://drive.google.com/drive/folders/1XtMsccaqu5as0yxiJCjPQdlqS3ap5kph). Simply download the whole drive at the root of your local github clone. (less than 1GB) 

## 3. Git structure. 
Once step 2 performed, you should achieve the following git structure: 
SEBASTIAN : Describe structure here. 
```
project
│   README.md
│   bert.ipynb
│   cooc.pkl""used for glove training""
|   helpers.py ""functions primarily used in ml_methods_embeddings.ipynb""
|   ml_methods_embeddings.ipynb
|   vocab.pkl""vocabulary that will be used in ml_methods_embeddings.ipynb""
└───folder
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───imported_embeddings""used in the ml_methods_embeddings.ipynb notbook""
|   |""4 embedding files""
|
└───neural_nets""used in the ml_methods_embeddings.ipynb notbook""
|   |layers2_nodes128_3cnn_filters100_size4_3_2.pth
|
└─── twitter-datasets
|    |train_pos.txt
|    |train_neg.txt
|    |train_pos_full.txt
|    |train_neg_full.txt
|    |test_data.txt

```

## 4. EMBEDDING AND ML 
An example of how the embedding and ML was trained is in the ml_methods_embeddin notbook with clear instructions how to proceed. I the helpers functions there is also the functionality to train your own GloVe embedding as well as import pre-trained models. You can create your own vocabulary and co-occurance matrix as well by running the files in the build_vocab folder if you want to use that instead and then replace the original vocab.pkl and cooc.pkl files with those instead.

## 5. BERT

Bert was entirely trained with the help of google notebook and drive. 
This is due to way better running time (GPU) and large files involved when we train models.  

We provide a file called jupyter.ipynb that along with a google colab drive link containing all the necessary files. 
It was created in a first place to be run with google colab. 
All paths refer to the following google [drive folder](https://drive.google.com/drive/folders/11-iqSDHChz9ihD_9gY5L3SKspiwuwyil?usp=sharing), with public access. Note this is not entirely the same drive as the one mentionned in section 2, as we did not want you to download by mistake too much data. 
All instructions contained here are moreover reminded in the first section of the jupyter notebook. 

Simply click the small arrow on the right of the title, "ML_Project2" and add it as a short link to your personal drive, at the /MyDrive/ level. 
It contains everything you need to run this google colab, create models, train and test them. Then you can run bert.ipynb in Google Colab as usual.

## 6. run.py

The run.py is a small piece of the bert.ipynb aimed to recreate exactly our submission to AI-Crowd. 
We provide together with it the vinai_models100k folder, containing the corresponding train model, previously dowloaded in section 2. The code is short and self-explainatory. 
You should be able to execute it even locally in a reasonable time. 

