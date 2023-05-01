import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Preprocessing text
def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    stopwords = set(['a', 'an', 'and', 'the', 'to', 'in', 'of', 'for', 'that', 'with', 'as', 'at', 'by', 'this', 'from'])
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

#2. Feature Extraction
def extract_features(texts):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    return features

#3. Similarity Calculation(Cosine-Similarity)
def calculate_similarity(features):
    similarity_matrix = cosine_similarity(features)
    text_names = [f"Text File {i}" for i in range(len(sample_files))]
    return similarity_matrix, text_names

#4. Threshold Setting
def threshold_similarity(similarity_matrix, threshold = 0.5):
    num_docs = similarity_matrix.shape[0]
    pairs = [(i, j) for i in range(num_docs) for j in range(i+1, num_docs)]
    scores = similarity_matrix[np.triu_indices(num_docs, k=1)]
    predicted = scores > threshold
    return pairs, predicted

#5. Driver Code 
if __name__ == '__main__':
    sample_files = [doc for doc in os.listdir() if doc.endswith('.txt')]

    #preprocessed data
    sample_contents = []
    for file in sample_files:
        with open(file, 'r', encoding='utf-8') as f:
            sample_contents.append(f.read())

    #sample_contents = [open(file).read() for file in sample_files]
    preprocessed_contents = [preprocess_text(content) for content in sample_contents]

    # extract features
    features = extract_features(preprocessed_contents)
    
    # calculate similarity
    import pandas as pd
    similarity_matrix, text_names = calculate_similarity(features)
    df = pd.DataFrame(similarity_matrix, columns = text_names, index = text_names)   #This will print a pandas DataFrame that shows the pairwise similarity scores between each pair of documents.
    print(df)
    
    # detect plagiarism
    import numpy as np
    import pandas as pd
    pairs, predicted = threshold_similarity(similarity_matrix, threshold=0.5)
    print(threshold_similarity)
    true_pairs = [0, 0, 0, 1, 1, 1]
    true_labels = np.array([pair in true_pairs for pair in pairs])
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = np.sum(true_labels & predicted)
    confusion_matrix[0, 1] = np.sum(~true_labels & predicted)
    confusion_matrix[1, 0] = np.sum(true_labels & ~predicted)
    confusion_matrix[1, 1] = np.sum(~true_labels & ~predicted)
    print(confusion_matrix)

  # compute accuracy rate
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)     #true-positive + true-negative / total samples
    print(f"Accuracy rate: {accuracy}")

 # print plagiarized pairs
    for pair, label in zip(pairs, predicted):
        if label == True:
            print(f"{text_names[pair[0]]} is plagiarized from {text_names[pair[1]]}")
   



    