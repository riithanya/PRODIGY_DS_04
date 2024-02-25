import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix


train_df = pd.read_csv('twitter_training1.csv', names=['Index', 'Topic', 'Sentiment', 'Tweet'], encoding='latin1', nrows=10000)
validation_df = pd.read_csv('twitter_validation1.csv', names=['Index', 'Topic', 'Sentiment', 'Tweet'], encoding='latin1')


train_df['Sentiment'].fillna(-1, inplace=True)


train_df['Tweet'].fillna('', inplace=True)


tfidf_vectorizer = TfidfVectorizer()


X_train = tfidf_vectorizer.fit_transform(train_df['Tweet'])
y_train = train_df['Sentiment']


svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)


X_validation = tfidf_vectorizer.transform(validation_df['Tweet'])
y_validation = validation_df['Sentiment']


y_pred = svm_classifier.predict(X_validation)


print("Classification Report:")
print(classification_report(y_validation, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, y_pred))

