first_cookie_project
==============================

The repository for the exercises of the different readings of the MLOPS course. The latter is mainly made of the different implementations related to the reproducibility and the deployment of a model. This github is made of the framework :

- The main framework of cookiecutter. In src/models is implemented a neural network for the classification of MNIST images. A lot of things have been added in order to answer the different exercises. 

- An additional file "exercises_from_lecture". This folder contains the additional implementations and files that are not related to the modifications of the main project (MNIST).

I explain in this readme all the changes made in each course. This is patched between the main project and the specific exercises in exercises_from_lecture.

Modifications
------------

> S1 :

- MNIST : adding the dependencies to conda. 

- Exercises from lecture : completed the notebooks about pytorch.

> S2

- MNIST : adding the code to github, creating the cookiecutter framework on the new repository, install flake8, black and isort (and run them on the repository) adding DVC in a google drive.

> S3 :

- MNIST : adding the Docker images and Hydra configuration to the model.

- Exercises from lecture : /

> S4 :

- MNIST : create wandb account and add it to the main project, web link : https://wandb.ai/loic-ker/MNIST.

- Exercises from lecture : debugging and creating profile for vae_mnist_working.py.

> S5 :

- MNIST : adding continuous integration (in the workflow of .github) for flake8, isort and pytest.

- Exercises from lecture : /

> S6 :

- MNIST : cloud setup, switch the data from google drive to a google cloud bracket, build an image on google cloud (in the cloudbuild.yaml).

- Exercises from lecture : /

> S7 :

- MNIST : implement a quantize model in src/models/model and then compare the time of execution on the src/models/quantize_test.py. We have two train models. The results show that we obtain a similar accuracy but the training is faster.

- Exercises from lecture : performance on the LFW dataset (distribute training), implement data_parallel.py (distribute data) (I have implemented here without gpu because I dont have one), implement computational complexity (quantize).

> S8 :

- MNIST : implement local and cloud deployment.

- Exercises from lecture : /

