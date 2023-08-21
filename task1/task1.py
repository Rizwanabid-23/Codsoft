import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load train data
train_data_path = 'D:/Codsoft/task1/datasets/train_data.txt'
with open(train_data_path, 'r', encoding='utf-8') as file:
    train_lines = file.readlines()

train_data = []
for line in train_lines:
    parts = line.strip().split(':::')
    if len(parts) == 4:
        train_data.append({'id': parts[0], 'title': parts[1], 'genre': parts[2], 'description': parts[3]})

train_df = pd.DataFrame(train_data)

# Preprocessing for train data
train_df.dropna(subset=['description', 'genre'], inplace=True)

# Split data
X_train = train_df['description']
y_train = train_df['genre']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Train the Naive Bayes Model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)

# Load test data
test_data_path = 'D:/Codsoft/task1/datasets/test_data.txt'
with open(test_data_path, 'r', encoding='utf-8') as file:
    test_lines = file.readlines()

test_data = []
for line in test_lines:
    parts = line.strip().split(':::')
    if len(parts) == 3:
        test_data.append({'id': parts[0], 'title': parts[1], 'description': parts[2]})

test_df = pd.DataFrame(test_data)

# Preprocessing for test data
test_df.dropna(subset=['description'], inplace=True)

# TF-IDF Vectorization for test data
X_test = test_df['description']
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions
y_pred = naive_bayes.predict(X_test_tfidf)

# Export predictions
test_df['predicted_genre'] = y_pred
test_df.to_csv('predicted_genres.txt', sep=':', index=False, header=False)
