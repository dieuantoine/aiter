#!/bin/bash

echo "Datasets creation"
python -m src.aiter_lab.data.cli create_ds

echo "Temporary hypotheses CSV initialization"
python -m src.aiter_lab.data.cli create_temp_csv