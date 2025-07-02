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

    ./sh/run_selection_app.sh

### Run the reference creation app

    ./sh/run_reference_app.sh

### Compute the evaluation scores
Once everything is set up, you can compute the evaluation scores using the following command:

    ./sh/run_scores_calculation.sh

This internally runs:

    python -m scripts.score_calculation --col [COL] [--overwrite OUTPUT_FILENAME]

Arguments:
- <code>--col</code> (required): The name of the column to reformulate (hyp or ref)

- <code>--overwrite</code> (optional): If omitted, a new results file will be created (auto-named). 
If a filename is provided, the new scores will be appended to that file.

The results will then be stored in the <code>data/</code> folder

## Metadata and Versioning

Each time a score file is generated, its associated metadata is automatically saved in <code>data/metadata.csv</code>.

This metadata includes:
- The version of the references
- The prompt version used
- The code version or scoring logic

This allows for transparent tracking and reproducibility of all evaluations.

### Updating a Component Version
If you want to update any component (e.g. prompt, reference, or logic) follow these steps:

1. Create a new file with the updated content and incremented version number in the filename.
For example, if you're updating the prompt <code>hyp_reformulation_prompt_4.txt</code>, create <code>hyp_reformulation_prompt_5.txt</code>
2. Update the <code>config/version.yaml</code> file accordingly to reflect the new version:
<code yaml>PROMPT_VERSION: 5</code>

This ensures that the system records and uses the correct version, and that the metadata stays consistent.



## Project Structure

The project is divided into several modules:

- <code>analysis/</code> for data and results analyses
- <code>src/data/</code> scripts to process raw data and prepare working datasets
- <code>src/annotation/</code> Streamlit applications for human annotation (requests selection and reference creation)
- <code>src/llm_api/</code> scripts for Mistral API calls for the metric
- <code>src/metric/</code> Metric implementation and score computation logic

## Author
- Antoine Dieu (ALT-EDIC) - dieu.antoine92@gmail.com