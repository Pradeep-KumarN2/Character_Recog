# Character Recognition Using Classical Machine Learning Techniques

## Overview

This repository contains an experimental character recognition workflow
built using image processing, feature extraction, and traditional
machine learning techniques.

The experiments are organized in the Jupyter notebook `Char_OCR.ipynb`,
while reusable image-preprocessing functions are provided in
`image_preprocessing.py`.

## Repository Contents

``` text
Character_Recog/
├── Char_OCR.ipynb
├── image_preprocessing.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── datasets/
│   └── .gitkeep
└── models/
    └── .gitkeep
```

## Methodology

The notebook explores multiple stages of a traditional computer-vision
and machine-learning workflow.

### Image preprocessing

Image-preprocessing operations are implemented in
`image_preprocessing.py` and used by the notebook.

### Feature extraction

The notebook includes experiments with:

-   Histogram of Oriented Gradients (HOG)
-   Scale-Invariant Feature Transform (SIFT)
-   HOG and SIFT feature extraction with and without shading-related
    processing
-   Bag of Visual Words (BoVW) using K-Means clustering

### Feature processing and dimensionality reduction

The experiments also include:

-   Feature scaling
-   Incremental Principal Component Analysis (Incremental PCA)

### Classification

The notebook evaluates traditional machine-learning classifiers,
including:

-   Support Vector Machine (SVM)
-   Random Forest
-   Gaussian Naive Bayes

Model evaluation and parameter-tuning experiments are also included in
the notebook.

## Requirements

Python dependencies are listed in `requirements.txt`.

Create and activate a virtual environment, then install the
dependencies:

``` bash
python -m venv .venv
```

On Windows PowerShell:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Install the packages:

``` bash
pip install -r requirements.txt
```

## Dataset

The dataset is not included in this repository.

The notebook refers to the following data locations:

-   `processed/maithili_new_data/`
-   `processed/resized_images/`
-   `processed/sift_shaded_128/`
-   `processed/hog_shaded_128/`

Obtain and organize the required dataset locally according to the paths
expected by the notebook before running the experiments. See
`data/README.md` for the dataset note.

Only use or redistribute datasets in accordance with their applicable
license and usage terms.

## Running the Notebook

Run Jupyter Notebook from the repository root so that the notebook can
locate `image_preprocessing.py`:

``` bash
jupyter notebook
```

Open `Char_OCR.ipynb` and execute the notebook cells in order.

## Generated Files

The `datasets/` and `models/` directories are placeholders for generated
feature datasets and trained model files. Generated artifacts are not
included in the repository by default.

## Notes

This repository documents an experimental workflow. Results may vary
with the dataset, preprocessing choices, feature representation, and
classifier configuration. The notebook should be consulted for the exact
experiment sequence and implementation details.
