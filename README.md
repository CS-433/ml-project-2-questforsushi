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

## 3. EMBEDDING AND ML 
SEBASTIAN : add here stuff about your files. Talk also about the file structure 

## 4. BERT

Bert was entirely trained with the help of google notebook and drive. 
This is due to way better running time (GPU) and large files involved when we train models.  

We provide a file called jupyter.ipynb that along with a google colab drive link containing all the necessary files. 
It was created in a first place to be run with google colab. 
All paths refer to the following google [drive folder](https://drive.google.com/drive/folders/11-iqSDHChz9ihD_9gY5L3SKspiwuwyil?usp=sharing), with public access. Note this is not entirely the same drive as the one mentionned in section 2, as we did not want you to download by mistake too much data. 
All instructions contained here are moreover reminded in the first section of the jupyter notebook. 

Simply click the small arrow on the right of the title, "ML_Project2" and add it as a short link to your personal drive, at the /MyDrive/ level. 
It contains everything you need to run this google colab, create models, train and test them. Then you can run bert.ipynb in Google Colab as usual.

## 5.run.py

The run.py is a small piece of the bert.ipynb aimed to recreate exactly our submission to AI-Crowd. 
We provide together with it the vinai_models100k folder, containing the corresponding train model, previously dowloaded in section 2. The code is short and self-explainatory. 
You should be able to execute it even locally in a reasonable time. 

