# How I Built an AI Mask Detector Application with just 200 Lines of Code
### I used an open source MLOps orchestration framework to build a production-ready ML image classifier system that detects mask wearing. It took me a few days, and just 200 lines of code. Here's how I did it.

In this post, I'm going to take you through how I built a production-ready ML system using [**MLRun**, the open source 
MLOps orchestration framework**](https://www.mlrun.org/), to easily orchestrate the heavy lifting of AI deployment. If 
your team is just starting on its ML operationalization journey, you might get the idea that you need an army of 
developers and endless resources to deploy an AI application. For this project, I used some new strategies and best 
practices that have emerged in the past year or so to tackle these MLOps challenges, so this example took me just a few 
days and 200 lines of code.

To focus this project around the challenges of operationalizing ML, I started with an 
[existing project from Kaggle](https://www.kaggle.com/notadithyabhat/face-mask-detector). This project is an image 
classification model that can detect whether a person in an image is wearing a mask or not. There are so many great 
ideas stuck in the lab, so I wanted to show how a truly useful - and even life saving - AI application can be deployed 
and start making impact very rapidly. While some countries may be living with a post-COVID-19 mindset at the moment, 
there are still millions of people around the world living through a pandemic. Far from being an academic data science 
exercise, this project could be so useful to help local officials and health organizations predict contagion spread in 
real time, and we'll see soon how quick it is to implement.

Taking the Kaggle project and wrapping it with MLRun enables us to version our data, track our experiments, distribute 
the training to run on multiple workers, possibly using GPUs, and finally serve our model on a very high-performance 
framework. All of these steps transform this little lab project to a real, production-ready, AI application.

# Mask Detection Demo
The following example demonstrates an end-to-end data science workflow.<br/>
We will start by grabbing a dataset from the web, train it in a multi-node multi-gpu fashion with tf on Horovod and 
deploy it as a real time serving function using Nuclio.
The model is trained on a dataset containing images of people with or without masks.<br/>
The model is then deployed to a Nuclio function allowing users to send an http request with an image and receive a 
response back with the probability that the person in the image is wearing a mask.

## Key Technologies
- [**Tensorflow-Keras**](https://www.tensorflow.org/api_docs/python/tf/keras) to train the model
- [**Horovod**](https://horovod.ai/) to run distributed training
- [**Nuclio**](https://nuclio.io/) to create a high-performance serverless Serving function
- [**MLRun**](https://www.mlrun.org/) to orchestrate the process

## Load Project
For this demo we're going to use **mlrun-kit** (Helm chart) which installs MLRun alongside a stack of tools (Nuclio, 
Jupyter, NFS...) on Kubernetes in one command, all preconfigured and integrated. The installation instructions can be 
found [here](https://docs.mlrun.org/en/latest/install.html). After installation we will open the Jupyter, create a new 
notebook, and use these commands to pull the code for this demo:
```jupyter
import mlrun
mlrun.load_project('./data/mask-detection', 'git://github.com/hedingber/mask-detection.git')
```

Then, open the [mask detection notebook](mask_detection.ipynb) and follow the instructions there.
