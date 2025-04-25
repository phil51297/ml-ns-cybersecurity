# Cybersecurity Threats Analysis

This project applies unsupervised machine learning techniques to analyze global cybersecurity threats from 2015 to 2024. Using **K-means clustering** and **anomaly detection**, the application identifies patterns in cyber attacks and potential outliers in the dataset.

## Features

- **Data Exploration and Visualization**: Detailed analysis of cybersecurity data with insightful visualizations  
- **K-means Clustering**: Identification of attack patterns based on multiple features  
- **PCA Dimensionality Reduction**: 2D visualization of high-dimensional cybersecurity data  
- **Anomaly Detection**: Identification of unusual cyber incidents using Isolation Forest  
- **Detailed Reporting**: Comprehensive analysis and recommendations based on the findings  

## Dataset

The application uses the **Global Cybersecurity Threats 2015–2024** dataset from Kaggle, which includes information about:

- Attack types and sources  
- Target industries  
- Financial impact  
- Number of affected users  
- Security vulnerabilities  
- Defense mechanisms  
- Resolution times  

## Installation

### Prerequisites

- Python 3.8+  
- `pip`  

### Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/phil51297/ml-ns-cybersecurity.git
    cd ml-ns-cybersecurity
    ```

2. Create and activate a virtual environment (recommended):

    ```bash
    python -m venv env
    ```

    **On Windows:**

    ```bash
    env\Scripts\activate
    ```

    **On macOS/Linux:**

    ```bash
    source env/bin/activate
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main analysis script:

```bash
python cybersecurity_analysis.py
```

This will:

- Download the dataset from Kaggle
- Perform exploratory data analysis
- Apply K-means clustering
- Perform anomaly detection
- Generate visualizations in the visualizations folder
- Create analysis reports in the reports folder
- Save processed data in the results folder

## Results
The analysis results include:

- Identification of 4 distinct cybersecurity attack clusters
- Detection of anomalous cyber incidents
- Visual representations of attack patterns
- Recommendations for improving cybersecurity defenses
