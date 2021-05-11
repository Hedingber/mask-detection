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
