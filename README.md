# TextSummarization-Transformer


BART-TextSummary-TF is a text summarization project utilizing the BART (Bidirectional and Auto-Regressive Transformers) model to generate high-quality summaries from lengthy text documents. This repository provides a complete pipeline for training, evaluating, and comparing summarization models using TensorFlow and the BART architecture.

Overview
---------
Text summarization is a natural language processing (NLP) task aimed at condensing long text documents into shorter, meaningful summaries. This project focuses on abstractive summarization, where the model generates new sentences that capture the essence of the original text, rather than simply selecting and reusing sentences.

The BART model, developed by Facebook AI, is particularly effective for sequence-to-sequence tasks such as summarization. This repository includes scripts and resources for training the BART model, evaluating its performance, and comparing it with an extractive summarization baseline.

Features
---------
- Abstractive Summarization with BART: Uses the BART model to generate summaries that are fluent and contextually accurate.
- Dataset Preparation: Scripts to download and preprocess the CNN/Daily Mail dataset.
- Model Training: Fine-tunes the BART model using TensorFlow with configurable parameters.
- Evaluation Metrics: Evaluates model performance using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L).
- Baseline Comparison: Compares BART-generated summaries with extractive summaries using the LsaSummarizer.

Getting Started
----------------
Prerequisites:
- Python 3.7+
- TensorFlow
- Hugging Face Transformers
- Sumy
- NLTK
- Scikit-learn

Installation:
1. Clone the repository:
   git clone https://github.com/yourusername/BART-TextSummary-TF.git
   cd BART-TextSummary-TF

2. Install the required dependencies:
   pip install -r requirements.txt

Dataset Preparation:
1. Run the dataset preparation script:
   python data_preparation.py
   This script downloads and preprocesses the CNN/Daily Mail dataset.

Training the Model:
1. Train the BART model:
   python train_model.py
   The training script fine-tunes the BART model on the dataset. Adjust training parameters as needed.

Evaluating the Model:
1. Evaluate the trained model:
   python evaluate_model.py
   This script computes ROUGE scores to assess the quality of generated summaries.

Running Baseline Comparison:
1. Compare with extractive summarization:
   python baseline_comparison.py
   This script uses the LsaSummarizer to generate extractive summaries and compares them with BART-generated summaries.

Code Overview
--------------
- data_preparation.py: Downloads and preprocesses the CNN/Daily Mail dataset.
- train_model.py: Fine-tunes the BART model on the dataset.
- evaluate_model.py: Evaluates model performance using ROUGE metrics.
- baseline_comparison.py: Compares BART-generated summaries with extractive summaries.
- requirements.txt: Lists Python dependencies.
- references.bib: Contains bibliographic references.

Results
---------
The results section includes ROUGE scores evaluating the BART model's performance. The comparison with extractive summarization provides insights into the effectiveness of the BART model in generating more coherent summaries.

Contributing
-------------
Contributions are welcome! Fork the repository and submit a pull request with your changes. For major modifications, discuss your proposal by opening an issue before creating a pull request.

License
--------
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
----------------
- Hugging Face Transformers: Provides the BART model and tools for NLP tasks.
- Sumy: Offers extractive summarization methods used for baseline comparison.
