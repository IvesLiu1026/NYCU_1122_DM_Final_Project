# Data Mining Final Project Report

## Preprocessing

### Behaviors
1. **Reading Behaviors Data**: The behaviors data is read from a TSV file (`behaviors.tsv`). This file contains user interactions with news articles.
2. **Filling Missing Values**: Any missing values in the `clicked_news` column are filled with a space (' ').
3. **Splitting Impressions**: The `impressions` column, which contains the news articles shown to users along with their click status, is split into individual elements.
4. **Mapping Users to Integers**: A mapping from user IDs to integer indices is created and saved to a file (`user2int.tsv`).

### News
1. **Reading News Data**: The news data is read from a TSV file (`news.tsv`). This file contains the news articles along with their metadata such as category, subcategory, title, and abstract.
2. **Filling Missing Values**: Any missing values in the `title_entities` and `abstract_entities` columns are filled with empty lists ('[]').
3. **Cleaning JSON Strings**: The JSON strings in the `title_entities` and `abstract_entities` columns are cleaned and parsed.
4. **Mapping Categories and Subcategories to Integers**: The categories and subcategories are mapped to integer indices.
5. **Tokenizing Titles and Abstracts**: The titles and abstracts are tokenized, and each token is mapped to an integer index. Additionally, entities in the titles and abstracts are also mapped to their respective integer indices.
6. **Saving Parsed News Data**: The parsed news data is saved to a file (`news_parsed.tsv`).

## Embeddings

### Word Embeddings
1. **Loading Pretrained Word Embeddings**: Pretrained word embeddings (e.g., GloVe) are loaded.
2. **Creating Word Embedding Matrix**: An embedding matrix is created where each row corresponds to a word and contains its embedding. Words not found in the pretrained embeddings are initialized with random values.
3. **Saving Word Embedding Matrix**: The embedding matrix is saved to a file (`pretrained_word_embedding.npy`).

### Entity Embeddings
1. **Loading Entity Embeddings**: Pretrained entity embeddings are loaded.
2. **Transforming Entity Embeddings**: The entity embeddings are transformed and saved to a file (`pretrained_entity_embedding.npy`).

## Model Selection

The model used in this solution is the **NRMS (Neural News Recommendation with Multi-Head Self-Attention)** model. NRMS is a deep learning-based model designed for news recommendation tasks.

### Model Architecture
- **Multi-Head Self-Attention**: This mechanism is used to capture the interactions between different parts of the news articles.
- **User Encoder**: Encodes the user’s clicked news history into a user embedding.
- **News Encoder**: Encodes the news articles into news embeddings.

## Hyperparameters

The hyperparameters used in this solution include:
- **Embedding Dimensions**: The dimensions of the word and entity embeddings.
- **Negative Sampling Ratio**: The ratio of negative samples to positive samples used during training.
- **Number of Words in Title/Abstract**: The maximum number of words considered from the title and abstract of each news article.
- **Batch Size**: The number of samples processed before the model’s internal parameters are updated.
- **Learning Rate**: The step size at each iteration while moving towards a minimum of the loss function.

## Implementation Details

### Data Preprocessing
The data preprocessing steps are implemented in the `data_preprocess.py` file. This script reads the raw data files, processes them, and saves the processed data files.

### Training
The training process is implemented in the `train.py` file. This script trains the NRMS model using the processed data and saves the trained model parameters.

### Evaluation
The evaluation process is implemented in the `evaluate.py` file. This script evaluates the trained model on the validation and test datasets and saves the predictions.

## Conclusion

This solution preprocesses the input data, trains the NRMS model, and evaluates it on the validation and test datasets. The results are saved and can be used for further analysis and reporting.

