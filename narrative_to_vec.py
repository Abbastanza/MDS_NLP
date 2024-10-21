from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import os


# read data
def read_data(folder_name, zip):
    data_path = os.path.join(folder_name, zip)
    data = pd.read_csv(data_path)
    return data

# tokenize a column
def tokenize_column(data, text_col, new_col):
    data[new_col] = data[text_col].apply(lambda x: x.split())
    return data

# train word2vec model on tokenized column
def train_word2vec_model(data, col_name, vector_size=150, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences=data[col_name], 
                     vector_size=vector_size, 
                     window=window, 
                     min_count=min_count, 
                     workers=workers)
    model.save("narrative_word2vec.model")

# Average word vectore and store as a new column
def add_column_average_genre_vector(data, model_path, col_name, vector_size=150):
    # load the trained model
    model = Word2Vec.load(model_path)
    # Create a dictionary of word embeddings
    word_embeddings = {word: model.wv[word] for word in model.wv.index_to_key}

    # Average word vector
    def average_genre_vector(word_list, word_embeddings, vector_size):
        # drop words not present in the embedding model
        valid_embeddings = [word_embeddings[word] for word in word_list if word in word_embeddings]
        if not valid_embeddings:
            return np.zeros(vector_size)  # return a zero vector if no valid genres
        # calculate mean vector
        mean_vector = np.mean(valid_embeddings, axis=0)
        return mean_vector

    # Apply to dataset
    data[col_name + '_vector'] = data[col_name].apply(
        lambda words: average_genre_vector(words, word_embeddings, vector_size))

    return data

# store as CSV
def store_data(data, path):
    data.to_csv(path, index=False)

# Main
def main():
    col_name = "narrative_tokenized"
    # read data
    data = read_data("data", "data_eda.zip")

    # tokenize the column
    data = tokenize_column(data, "narrative_prep", col_name)

    # train a Word2Vec model 
    train_word2vec_model(data, col_name=col_name)

    # Add a new column with the average vector
    data = add_column_average_genre_vector(data, 
                                           model_path="narrative_word2vec.model", 
                                           col_name=col_name)

    # store the data as a new CSV file
    store_data(data, "data/data_narrative_vector.csv")

    print("Done!")

if __name__ == "__main__":
    main()