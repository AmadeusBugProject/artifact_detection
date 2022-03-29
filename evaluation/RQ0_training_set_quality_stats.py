import pandas as pd
from datasets.constants import LANGUAGES
from file_anchor import root_dir

PATH = root_dir() + 'evaluation/out/dataset_stats/'

language_labels = {
    'cpp': 'C++',
    'java': 'Java',
    'javascript': 'JavaScript',
    'php': 'PHP',
    'python': 'Python',
}


def main():
    wrong_artifacts = {}
    wrong_texts = {}
    for lang in LANGUAGES:
        df_artifact = pd.read_csv(PATH+lang+"_training_set_artifact_sample.csv")
        df_text = pd.read_csv(PATH + lang + "_training_set_text_sample.csv")
        wrong_artifacts[language_labels[lang]] = len(df_artifact[df_artifact['target'] == 1])
        wrong_texts[language_labels[lang]] = len(df_artifact[df_text['target'] == 0])

    df = pd.DataFrame.from_dict([wrong_artifacts, wrong_texts])
    df = df.rename(index={0: "Artifacts", 1: "Natural language"})
    with open(PATH+'training_set_quality_stats.tex', 'w') as f:
        f.write(df.to_latex())


if __name__ == "__main__":
    main()