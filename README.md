87% of data science projects never make it into production 
[(VentureBeat)](https://venturebeat.com/2019/07/19/why-do-87-of-data-science-projects-never-make-it-into-production/). 
The reasons are varying from absence of data to poor management expressed as communication problems,
lack of clear goals and solving the wrong problems. One thing that is constantly mentioned as a one of the main reasons is the 
underestimation of the operations involved in taking a data science project to production, or in one word MLOps.
(Add some quote/number from other article on expectations vs reality for time spent on MLOps)
In this article I'm gonna show how MLRun - an open source MLOps framework can help tackle these challenges

The MLOps phase is usually coming after you've built a draft for your model, to simulate that we'll use an existing 
project from [Kaggle](https://www.kaggle.com/). I wanted to have an interesting real life problem, so the topic I chose 
is (how surprisingly) The Coronavirus - although vaccination are already here and some countries have big portions of 
their population vaccinated, it looks like there's still more time until we'll get rid the Masks, therefore we'll use 
[this project](https://www.kaggle.com/notadithyabhat/face-mask-detector/execution) which is trying to train a model to
detect whether the people in the image are wearing mask or not.

For the this demo we're going to use the mlrun-kit - an helm chart enabling you to use one command to install a stack of
tools, all preconfigured and integrated, that enables a lot of the capabilities of MLRun, the installation instructions
can be found [here](https://docs.mlrun.org/en/latest/install.html). After installation we will open the jupyter, create 
a new notebook, and use these commands to pull the code for this demo:
```jupyter
import mlrun
mlrun.load_project('./data/mask-detection', 'git://github.com/hedingber/mask-detection.git')
```

To move on open the [mask detection notebook](mask_detection.ipynb) and follow the instructions there
