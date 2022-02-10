# artifact_detection
A tool for NLP tasks on textual bug reports.
Automated classification of text into natural language (e.g. English in the contained datasets) , and non-natural language text portions (e.g. stack traces, code snippets, log outputs, file listings, urls,) on a line by line basis.

This repo contains the Python implementation of a machine learning classifier model, basic scripts for automated trainingset creation from GitHub issue tickets, a sample dataset sourced from 101 Java projects hosted on GitHub, and a scikit-learn transformer that wraps the pretrained model to be used as preprocessing step in a scikit-learn pipeline.

More information can be found in the publication:
> Thomas Hirsch and Birgit Hofer: ["Identifying non-natural language artifacts in bug reports"](https://doi.org/10.1109/ASEW52652.2021.00046), 2nd International Workshop on Software Engineering Automation: A Natural Language Perspective (NLP-SEA) - 36th IEEE/ACM International Conference on Automated Software Engineering Workshops (ASEW), 2021, pp. 191-197, doi: 10.1109/ASEW52652.2021.00046. [arvix](https://arxiv.org/abs/2110.01336)

## Model
The classifier model is a scikit-learn [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) (Support Vector Classifier).
Implementation of the pipleine can be found in [model_training.py](artifact_detection_model/model_training.py).
Most notable preprocessing and feature creation step of this pipeline is in [SpecialCharacterToWords.py](artifact_detection_model/SpecialCharacterToWords.py).

## Training
[RUN_train_model.py](artifact_detection_model/RUN_train_model.py) trains and evalutates a new model on the included [datasets](datasets), and produces a joblib dump of that can be imported elsewhere.
The report and dump files will be created in [artifact_detection_model/out](artifact_detection_model/out).

## GitHub Java dataset
The trainingset is created from [issue tickets](datasets/training_set_bug_reports.csv.zip) and [.md documentation files](datasets/documentation_set.csv.zip) of 101 Java projects hosted on GitHub.
The [testset](datasets/test_set_bug_reports.csv.zip) also originates from the same Java projects.
The [validation set 1](datasets/validation_set_researcher_1.csv.zip) and [validation set 2](datasets/validation_set_researcher_2.csv.zip) are a random sample from the testset that were manually annotated.

Licences of each of the included projects can be found in [projects](datasets/licences).

The above listed datasets are zip compressed csv and can be loaded via `pandas.read_csv('...', compression='zip')`.
Automated trainingset labeling is done in [dataset_creation.py](artifact_detection_model/dataset_creation.py) and [regex_cleanup.py](artifact_detection_model/regex_cleanup.py)

## NLoN dataset
The [NLoN](https://github.com/M3SOulu/NLoN) dataset was created by Mäntylä M. V., Calefato F., Claes M, "Natural Language or Not (NLoN) - A Package for Software Engineering Text Analysis Pipeline", The 15th International Conference on Mining Software Repositories (MSR 2018), May 28--29, 2017, Gothenburg, Sweden, pp. 1-5 https://mmantyla.github.io//2018_Mantyla_MSR_natural-language-nlon.pdf.

## Evaluation
All evaluation scripts can be found in [evaluation](evaluation). This includes interrater agreement investigation into validation and NLoN datasets, creation of learning curve of our model, and cross evaluation of this model and the NLoN model.

## Use this in a NLP project
If you want to use this model in another project, create a joblib dump of a pretrained model by running [RUN_train_model.py](artifact_detection_model/RUN_train_model.py.py), and then copy the [artifact_detection_model](artifact_detection_model) directory over to the other project.

[ArtifactRemoverTransformer.py](artifact_detection_model/transformer/ArtifactRemoverTransformer.py) is a scikitlearn transformer that removes or replaces artifacts from textual bug reports ready for use.

# Acknowledgment
The work has been funded by the Austrian Science Fund (FWF): P 32653-N (Automated Debugging in Use).

