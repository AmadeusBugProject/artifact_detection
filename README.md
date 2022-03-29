# artifact_detection
A tool for NLP tasks on textual bug reports.
Automated classification of text into natural language (e.g. English in the contained datasets), and non-natural language text portions (e.g. stack traces, code snippets, log outputs, file listings, urls,) on a line by line basis.

## About
This repo contains the Python implementation of a machine learning classifier model, basic scripts for automated trainingset creation from GitHub issue tickets.
Further, a scikit-learn transformer implementation wrapping pretrained models ready to be used as preprocessing step.
Datasets consist of issue tickets and documentation files mined from C++, Java, JavaScript, PHP, and Python projects hosted on GitHub.

## Publications
- The latest, most extensive paper (in review) and corresponding datasets and implementations:
> Thomas Hirsch and Birgit Hofer: ["Detecting non-natural language artifacts for de-noising bug reports"](TODO)
> 
> [Zenodo](https://zenodo.org/record/6393129)
> 
> [GitHub Release](https://github.com/AmadeusBugProject/artifact_detection/)

- The original workshop paper and corresponding datasets and implementations:
> Thomas Hirsch and Birgit Hofer: ["Identifying non-natural language artifacts in bug reports"](https://doi.org/10.1109/ASEW52652.2021.00046), 2nd International Workshop on Software Engineering Automation: A Natural Language Perspective (NLP-SEA) - 36th IEEE/ACM International Conference on Automated Software Engineering Workshops (ASEW), 2021, pp. 191-197, doi: 10.1109/ASEW52652.2021.00046. [arvix](https://arxiv.org/abs/2110.01336)
> 
> [Zenodo](https://zenodo.org/record/5519503)
> 
> [GitHub Release](https://github.com/AmadeusBugProject/artifact_detection/releases/tag/v1.1)

# Preliminary steps
## Data
Datasets and pretrained models are not contained in this git repository due to their size, they are hosted on [Zenodo](https://zenodo.org/record/6393129).
Download the files and move them to the corersponding locations, language datasets go to [datasets](datasets), and pretrained models go to [artifact_detection_model/out](artifact_detection_model/out).

## Conda environment
[Conda environment file](conda.yml).

# Tool usage
Please download all required data as described in [Preliminary steps](#preliminary-steps) above.
You can use the pretrained models in [artifact_detection_model/out](artifact_detection_model/out), or train a new model using [RUN_train_model.py](artifact_detection_model/RUN_train_model.py).

A scikitlearn transformer for wrapping pretrained models, is provided here: [ArtifactRemoverTransformer.py](artifact_detection_model/transformer/ArtifactRemoverTransformer.py)
This transformer accepts full bug reports as inputs and removes non-natural artifacts from these reports.

Example:
```python
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from artifact_detection_model.transformer.ArtifactRemoverTransformer import ArtifactRemoverTransformer, SIMPLE
from file_anchor import root_dir

train_x, train_y, test_x, test_y = [...] # array of issue tickets
artifact_classifier = joblib.load(root_dir() + 'artifact_detection_model/out/' + 'some_model.joblib')
pipeline = Pipeline([('artifactspred', ArtifactRemoverTransformer(artifact_classifier)),
                     ('vect', CountVectorizer()),
                     ('clf', LinearSVC())])
pipeline.fit(train_x, train_y)
y_predicted = pipeline.predict(test_x)
```

# Details and Reproduction of our experiments
## Model
The classifier model is a scikit-learn [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) (Support Vector Classifier).
Implementation of the pipeleine can be found in [model_training.py](artifact_detection_model/model_training.py).
Most notable preprocessing and feature creation step of this pipeline is in [SpecialCharacterToWords.py](artifact_detection_model/SpecialCharacterToWords.py).

## Experiments and evaluations
All evaluation scripts can be found in [evaluation](evaluation). 
The scripts names indicate the corresponding research question of our journal paper, see [Publications](#publications).
The outputs of all these scripts can be found in [evaluation/out](evaluation/out).
These folders contain the all of our results and preliminary data used in our evaluation.
When re-evaluating from scratch, these scripts are supposed to be run in alphabetical order as they use data produced by their preciding evaluation scripts.

# Datasets
After following the [preliminary steps](#preliminary-steps), all datasets are located in [datasets](datasets).
The data is stored as zip-compressed csv's.
Please have a look at our journal paper (see [Publications](#publications)) for details on our dataset creation process.
- `[language]_all_issues.csv.zip` contains all mined issue tickets for this language's projects.
- `[language]_all_documentation.csv.zip` contains all mined documentation files for this language's projects.
- `[language]_validation_issues.csv.zip` contains each 250 issue tickets that were randomly sampled from all mined issue tickets.
- `[language]_reseracher_[1|2]_manually_labeled_validation_set.csv.zip` contains the manually annotated validation sets from both researchers from the above 250 issue tickets.
- `[language]_training_issues.csv.zip` contains the data used in training, that is issue tickets containing "```" markdown code blocks (excluding those in validation sets), and all documentation files.

The validation sets contain the researchers' manual annotation, `0` labels lines considered non-natural language artifacts, `1` labels lines considered natural language.

The projects from which the corresponding datasets originate from are listed here:
- [C++](githubMiner/json_dump/cpp.txt)
- [Java](githubMiner/json_dump/java.txt)
- [JavaScript](githubMiner/json_dump/javascript.txt)
- [PHP](githubMiner/json_dump/php.txt)
- [Python](githubMiner/json_dump/python.txt)


## Dataset creation
Data was mined from GitHub repositories of above linked projects.
[RUN_github_issue_ticke_mining.py](githubMiner/RUN_github_issue_ticke_mining.py) reads the projects list and stores the json replies from GitHub API.
[RUN_create_training_and_validation_sets.py](githubMiner/RUN_create_training_and_validation_sets.py) then consolidates the json API responses into Pandas DataFrames stored in [datasets](datasets), and performs the training / validation split of issue tickets.
The original json dumps are not included due to their size, and duplication, as the required data is provided as zipped csv's in [datasets](datasets).


# Acknowledgment
The work has been funded by the Austrian Science Fund (FWF): P 32653-N (Automated Debugging in Use).

