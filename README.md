MLops exercises (project)
==============================

The [MLops exercises](https://github.com/SkafteNicki/dtu_mlops) applied to a project on the MNIST fashion dataset. 

### Day 1 

Day one was about PyTorch in general, the application to the project are the definition of the python files

1. `model.py`, which defines the **Neural Network** for the dataset.  
   - It can be found [here](/src/models/model.py)
3. `train_model.py`, which **trains** the Neural Network and saves the corresponding weights.  
   - It is used by simply running it in a python interpreter.
   - It can be found [here](/src/models/train_model.py)

### Day 2

Day two was about code organization and setup. To initialize the the project I used `coockiecutter`. The files updated and created for that are 

1. `make_dataset.py`, which saves the MNIST (raw) dataset in the *data/preprocessed* folder as **training** and **test** set.  
   - It is used by simply running it in a python interpreter. 
   - It can be found [here](/src/data/make_dataset.py)
2. predict_model.py`, which uses the saved model to **predict** the class of a given image or several images.  
   - It is used by simply running it in a python interpreter.
   - It can be found [here](/src/models/predict_model.py).
3. `visualize.py`, which uses [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to visualize the extracted features of the model.  
   - It is used by simply running it in a python interpreter. 
   - It can be found [here](/src/visualization/visualize.py)
4. `requirements.txt`, which keeps track of the packages used for the project. 

Now after setting up the project the focus was on following a nice setup of code and comments, for that I used the flake8 package, more specifically the [black](https://github.com/psf/black) module was used to do this automatically. 

Finally the module [isort](https://github.com/PyCQA/isort) was used to setup how the packages are imported, again so the structure is nicely. 

### Day 3

Day three was about debugging the code and visualizing part of that. 





Project Organization
------------

    ????????? LICENSE
    ????????? Makefile           <- Makefile with commands like `make data` or `make train`
    ????????? README.md          <- The top-level README for developers using this project.
    ????????? data
    ??????? ????????? external       <- Data from third party sources.
    ??????? ????????? interim        <- Intermediate data that has been transformed.
    ??????? ????????? processed      <- The final, canonical data sets for modeling.
    ??????? ????????? raw            <- The original, immutable data dump.
    ???
    ????????? docs               <- A default Sphinx project; see sphinx-doc.org for details
    ???
    ????????? models             <- Trained and serialized models, model predictions, or model summaries
    ???
    ????????? notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    ???                         the creator's initials, and a short `-` delimited description, e.g.
    ???                         `1.0-jqp-initial-data-exploration`.
    ???
    ????????? references         <- Data dictionaries, manuals, and all other explanatory materials.
    ???
    ????????? reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    ??????? ????????? figures        <- Generated graphics and figures to be used in reporting
    ???
    ????????? requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    ???                         generated with `pip freeze > requirements.txt`
    ???
    ????????? setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ????????? src                <- Source code for use in this project.
    ??????? ????????? __init__.py    <- Makes src a Python module
    ???   ???
    ??????? ????????? data           <- Scripts to download or generate data
    ??????? ??????? ????????? make_dataset.py
    ???   ???
    ??????? ????????? features       <- Scripts to turn raw data into features for modeling
    ??????? ??????? ????????? build_features.py
    ???   ???
    ??????? ????????? models         <- Scripts to train models and then use trained models to make
    ???   ???   ???                 predictions
    ??????? ??????? ????????? predict_model.py
    ??????? ??????? ????????? train_model.py
    ???   ???
    ??????? ????????? visualization  <- Scripts to create exploratory and results oriented visualizations
    ???????     ????????? visualize.py
    ???
    ????????? tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>