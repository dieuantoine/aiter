# AITER-LAB
AITER-LAB is a tool designed to test and benchmark the [AITER metric](https://github.com/dieuantoine/aiter-metric).

The project aims to apply this metric to the [comparIA](https://huggingface.co/datasets/ministere-culture/comparia-conversations) and the [MKQA](https://huggingface.co/datasets/apple/mkqa) datasets and analyze the results in a structured and reproducible way.

## Installation

1. Clone the Git repository

```bash
git clone https://github.com/dieuantoine/aiter.git
cd aiter
```

2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Add secrets in a `.env` file

Create a `.env` file at the root of the project and add your `MISTRAL_API_KEY`, `GEMINI_API_KEY` and `HF_TOKEN`.

## Usage

This repository contains several utilities and scripts serving different purposes: data preparation, annotation, metric evaluation, datasets analysis and results analysis.

Below is a typical example workflow illustrating how to generate a dataset and apply the AITER metric, step by step.

<details><summary>Example Worlflow</summary>

### Set the correct version in the version file

Modify the `config/version.yaml` file with the correct settings. 

### Ingest and pre-process the data

Preprocess the data and upload the datasets to Hugging Face

```bash
sh scripts/0_run_datasets_creation.sh
```

### Run the requests selection app

```bash
sh scripts/1_run_selection_app.sh
```

### Create the hypotheses dataset (only for MKQA)

- Create the hypotheses csv file

```bash
sh scripts/1bis_create_temp_csv.sh
```

- Complete the csv file with models responses

- When all the hypotheses have been created, push the dataset to HuggingFace

```bash
sh scripts/1ter_push_temp_csv.sh
```

### Run the reference creation app

```bash
sh scripts/2_run_reference_app.sh
```

### Compute the evaluation scores
Once everything is set up, you can compute the evaluation scores using the following command:

```bash
sh scripts/3_run_scores_calculation.sh
```

The results will then be stored in the `data/results/` folder

</details>

## Metadata and Versioning

Each time a score file is generated, its associated metadata is automatically saved in `data/results/metadata.csv`.

This metadata includes the version used, the language, and the execution time of the process, allowing transparent tracking and reproducibility of all evaluations.

## Repository Structure

The project is divided into several modules:

```
aiter/
├── apps/                   # Streamlit applications for human annotation (request selection and reference creation)
├── config/                 # Configuration files and default project settings
├── data/                   # Datasets, scoring outputs, and analysis files
├── scripts/                # Shell scripts for running end-to-end workflows
└── src/
    └── aiter_lab/
        ├── analysis/       # Tools for analyzing datasets and evaluation results
        ├── config /        # Configuration scripts   
        ├── data/           # Scripts to process raw data and prepare working datasets
        ├── metric/         # Logic to run the AITER scorer on the processed data
        └── utils           # Utils functions
```

## Authors

[Antoine Dieu](mailto:dieu.antoine92@gmail.com) @[ALT-EDIC](https://www.alt-edic.eu/)