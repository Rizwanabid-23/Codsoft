
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('D:/Codsoft/task3/datasets/spam.csv', encoding='latin-1') 
X = data['v1']
y = data['v2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_tfidf, y_train)
y_pred_nb = naive_bayes.predict(X_test_tfidf)

print("Naive Bayes:")
print(confusion_matrix(y_test, y_pred_nb))
with open('classification_report1.txt', 'w', encoding='utf-8') as report_file:
    report = classification_report(y_test, y_pred_nb)
    report_file.write(report)


