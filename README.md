# Artwork Identification Using Fine-Tuned ResNet50 and FAISS Index

## Abstract
This project implements a system to identify artworks from a dataset using a fine-tuned ResNet50 model and a FAISS index for similarity search. By training on a curated set of labeled images, the system allows users to query an image and retrieve the most similar artworks along with their metadata. The approach combines supervised fine-tuning with unsupervised feature indexing to achieve high accuracy and fast retrieval.
I originally began as a fine-tuning project, by means of things like layer freezing, varying iteration amounts, varying validation and testing data sets, and variance in labeling and batch sizing. Eventually, I realized that, what might be more effecitve, was to absolutely broaden how large my dataset was, and instead, do something far more computationally efficient, and far more effective for the results that I wanted. So, after probably over 150 hours of K80 training on the cluster, I threw it all away (sort of), and decided to focus on similarity searching using the pretrained weights of the already so powerful ResNet50. 
Implementing FAISS, I was able to get amazing results, easily broaden my dataset, and complete computations on my computer, instead of on the cluster (slower, of course). So, this is the project. I'll walk you through the entire chronology of the project, and despite there only being something like 7 notebooks in this repo, I can assure you, there are countless iterations of my result fetching. 

## Overview

### What is the problem?
The project addresses the problem of identifying artworks based on visual features. Given a query image, the goal is to retrieve similar artworks from my database, which is made up of 60,000 artworks from WikiArt, and 110,000 artworks from the National Galery of art, enabling applications such as artwork cataloging, provenance tracking, and content-based image retrieval.
The intention was to allow a user to upload a photo of any artwork (including their own!), and immediately see the results, in the form of the most similar image in the dataset, the artist, title, and genre as well.

### Why is this problem interesting?
Art identification has wide-ranging applications in cultural preservation, museum management, and education. Automating this process can reduce manual labor, improve accessibility, and enhance user experience by providing contextual information about artworks in real time. The idea came to me while I was attempting to figure out if an artpiece that a friend had was a replica or not. Now, it was in a dorm room, so that was a clear tell, but I was wondering if there was a way to 
figure out exactly what piece I was looking at, no matter where I was. Google Reverse Image Search allowed me to do so, but with a lot more hoops to jump through. So, I decided to solve this.

### What is the approach to tackle the problem?
My solution leveraged a combination of a pre-trained ResNet50 model fine-tuned on artwork datasets for feature extraction and FAISS for efficient similarity search. The system processes a query image, extracts its features, and compares them against a pre-built FAISS index of artwork features to find the top matches. In the end, I was able to get my fine-tuned model up to about 88% accuracy, however, with similarity search, I was able to achieve over 98%, with JUST the pretrained ResNet50. This was excellent. As such, these are the results that I have provided, however, in the repo, there exist the fine-tuning files that I used as well. However, the formally named file "ResNet_Image_Retrieval.ipynb" is where my best results lie.

### What is the rationale behind the proposed approach?
ResNet50 has proven to be extremely effective for feature extraction in image classification tasks, while FAISS is optimized for high-dimensional vector similarity search. Together, they were able to provide me with a scalable and efficient solution for my problem. 

### Key components and results
- **Dataset preparation:** Aggregating and cleaning data from multiple sources to create a labeled dataset of artwork images and metadata. This took an extraordinary amount of time, as combining two datasets with hundreds of thousands of rows, and tens of thousands of null values in random spots, was a tall task, frought with tribulations.
- **Feature extraction and indexing:** Using the pretrained model to extract features and indexing them with FAISS for fast retrieval.
- **Performance:** My system achieves near exact accuracy in identifying artworks, with visualizations showcasing the top matches for each query.

### Limitations
- Relies on (in terms of purely identification) artpiece being within dataset. However, when inputting an image that is not, genre accuracy is performant as well.
- For efficiency, locally downloaded dataset is required.

## Experiment Setup

### Dataset
- **Source:** WikiArt, National Gallery of Art
- **Statistics:** ~170,000 unique artworks and corresponding metadata, filtered for sufficient samples per class

### Implementation
- **Model:** ResNet50 pre-trained on ImageNet.
- **Fine-tuning parameters:** Learning rate of 0.001, Adam optimizer, and cross-entropy loss.
- **Feature extraction:** Final layer features normalized and indexed using FAISS.
- **Hardware:** NVIDIA K80 GPU, 12 CPU, 128 GB RAM.

### Model architecture
- For Basic Model (which was used for concluded ouputs) - ResNet50 pre-trained on ImageNet.
- For Fine-Tuned model - ResNet50 with its fully connected layer replaced by a new classification layer matching the number of artwork classes. Initial layers are frozen during fine-tuning.

## Experiment Results

### Main results
- **Training and validation accuracy:** Achieved an accuracy of >98% when tested on **randomly rotated** and **blurred** images. >99% top 5 accuracy.
- **Loss reduction on Fine-Tuned Model:** Training and validation loss consistently decreased, indicating effective learning. However, this never achieved as effective results as FAISS.

## Discussion
The results demonstrate the effectiveness of combining fine-tuned ResNet50 and FAISS for artwork identification. I think that the next step would be to attempt fine-tuning even further, but instead of simply fine-tuning on the images in the dataset, fine tune on genres. I believe that the erason that I ran into so many issues when I was fine-tuning my model was because it was overfitting so ahrd, which is apparently common when training on very diverse ranges of images. In fact, discovering this was when I started to think more about FAISS. So, instead, I think my original plan would be far more effective if done on genre instead of the images themselves. 

## Conclusion
My project successfully developed a system for artwork identification that combines supervised learning and unsupervised feature indexing. It offers a scalable and efficient solution with potential applications in museums, galleries, and any cultural institutions, including individuals simply interested in artwork observation.

## References
- [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
- [FAISS Library](https://github.com/facebookresearch/faiss)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [WikiArt DataSet used](https://www.kaggle.com/datasets/steubk/wikiart)
- [National Gallery of Art DataSet used](https://www.nga.gov/open-access-images/open-data.html)
- [The Website that is in Progress](https://whatartpiece.art/)


# How to Use - ResNet_Image_Retrieval.ipynb

## Dataset
- The first order of business is ensuring that you download the dataset. For the sake of testing, you can simply set up a function that refers to the WikiArt dataset on something like HuggingFace.

## FAISS
- Next, when you run update directory references within the notebook, you should be able to run the notebook.
- Because you will not have a downloaded features file, it will use the pre-trained weights of ResNet50 to extract the features, and save them into a .bin file. This only needs to be done once, after features are extracted, it will use that same feature file for whatever art inputted.

## Running it
- Simply run the file.
- In its current state, it will go through a testing iteration of 1,000 test images, where it will use a gaussian distributed amount of blur and rotation, and then attempt to predict what artpiece it is.
- If you want to input your own image, simply comment out the section with the 1,000 rotated and blurred for loop'd pictures, and uncomment the section that I currently have, titled 'Section 7'.
- Refer to your image in the input image array, and in about 5 seconds flat, after running, you'll have it identified!

# >98% accuracy on rotated and blurred images, >99% TOP 5 accuracy on rotated and blurred images

