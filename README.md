# MDS_NLP - Consumer Complaints Classification

<!--  1. **Preprocessing**: Cleaned and tokenized the text, removed stopwords, and lemmatized the narratives.
2. **Exploratory Data Analysis**: Visualized product distribution, narrative lengths, and word clouds per category.
3. **Feature Engineering**: Used TF-IDF, GloVe embeddings, and Word2Vec for feature extraction.
4. **Modeling**: Tested multiple classifiers (Logistic Regression, Random Forest, XGBoost, Naive Bayes, SVM) with TF-IDF features.
5. **Evaluation**: Reported classification performance using accuracy and classification reports.-->

Research question:
How to develop an *automatic text classification model* that assigns customer complaints to the correct product categories based on their content?\
Authors: Alam Jasia, Hubweber Michaela, Schumich Kathrin, Ye Florian

## Introduction

This project focuses on developing an automatic text classification model that assigns customer complaints to the correct product categories based on their content. By accurately categorizing complaints, companies can address customer issues more efficiently and improve their products and services.

## Dataset

We used the **Consumer Complaints** dataset from Kaggle, which contains **162,421** entries. Each entry includes:

- **narrative**: The text of the consumer's complaint.
- **product**: The category of the product related to the complaint. Categories include:
  - Credit Reporting
  - Debt Collection
  - Mortgages and Loans
  - Credit Card
  - Retail Banking

The dataset is readily available for download from [Kaggle's website](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp).

## Installation

To set up the project locally:

1. **Download the dataset**:
   - Place the `raw.zip` file into the `data` folder.
2. **Install the required libraries** by running:

   ```bash
   pip install -r requirements.txt
   ```

## Libraries Used

- **scikit-learn** (`sklearn`): For machine learning algorithms.
- **spaCy**: For natural language processing tasks.
- **PyTorch** (`torch`): For building neural network models.
- **Gensim**: For working with Word2Vec embeddings.

## Preprocessing

We prepared the text data through the following steps:

- **Cleaning**: Removed unwanted characters and symbols.
- **Tokenization**: Split text into individual words.
- **Stopword Removal**: Removed common words that do not contribute much meaning (e.g., "the", "and").
- **Lemmatization**: Converted words to their base form (e.g., "running" to "run").

## Exploratory Data Analysis

We explored the dataset to understand its characteristics:

- **Product Distribution**: Visualized the number of complaints per product category.
- **Narrative Lengths**: Analyzed the length of complaint texts.
- **Word Clouds**: Generated word clouds for each category to highlight common words.

## Feature Engineering

We transformed the textual data into numerical features using:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Captures the importance of words in documents.
- **GloVe Embeddings**: Pre-trained word embeddings that capture semantic relationships.
- **Word2Vec**: Generated our own word embeddings from the dataset.

## Modeling

We tested several models to find the best performance:

### Classical Machine Learning Models

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

These models were trained using both TF-IDF and Word2Vec features. Hyperparameter tuning was performed for each model.

### Neural Network Models

- **Multilayer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**
- **Recurrent Neural Network (RNN)**
- **Bidirectional Long Short-Term Memory (Bi-LSTM)**

Neural networks were trained using Word2Vec features.

## Evaluation

We evaluated model performance using accuracy scores and classification reports.

- **TF-IDF Features**:
  - Classical models achieved around **70-75% accuracy**.
- **Word2Vec Features**:
  - Classical models achieved around **80-85% accuracy**.
  - Neural network models also achieved **80-85% accuracy**.

## Conclusion

Using Word2Vec features improved the accuracy of our models compared to TF-IDF features. Both classical machine learning models and neural networks performed similarly when using Word2Vec embeddings.

## Usage

Note: The models are too large and therefore not included in the Github repository. However, they are created and saved when the notebooks are executed.

To use the classification model:

1. Ensure all **dependencies are installed**.
2. Run the **preprocessing script** to prepare the data.
3. Execute the **model training scripts** for training the desired model.
4. Use the **trained model** to classify new consumer complaints.
