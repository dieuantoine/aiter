# AITER
AITER is a tool designed to introduce a new evaluation metric for large language models (LLMs).
The project also aims to apply this custom metric to the [comparIA dataset](https://huggingface.co/datasets/ministere-culture/comparia-conversations) and analyze the results in a structured and reproducible way.

## Installation

### 1. Clone the Git repository
    
    git clone https://github.com/dieuantoine/aiter.git
    cd aiter

### 2. Install the dependencies

    pip install -r requirements.txt

### 3. Add secrets in a <code>.env</code> file

Create a <code>.env</code> file at the root of the project and add <code>MISTRAL_API_KEY</code> and <code>HF_TOKEN</code>.

## Usage

### Pre-process the data

    python -m scripts.prepare_data

### Run the requests selection app

    streamlit run src/annotation/selection/app.py

### Run the reference creation app

    streamlit run src/annotation/reference_creation/app.py

### Compute the evaluation scores
Once everything is set up, you can compute the evaluation scores using the following command:

    python -m scripts.score_calculation --col [COL] --version [VERSION]
Arguments:
- <code>--col</code>: The name of the column to reformulate (hyp or ref)

- <code>--version</code>: The version of the scoring metric or evaluation logic used (will be in the name of the results file)

The results will then be stored in the <code>data/</code> folder


## Project Structure

The project is divided into several modules:

- <code>analysis/</code> for data and results analyses
- <code>src/data/</code> scripts to process raw data and prepare working datasets
- <code>src/annotation/</code> Streamlit applications for human annotation (requests selection and reference creation)
- <code>src/llm_api/</code> scripts for Mistral API calls for the metric
- <code>src/metric/</code> Metric implementation and score computation logic

## Author
- Antoine Dieu (ALT-EDIC) - dieu.antoine92@gmail.com