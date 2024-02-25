import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


train_df = pd.read_csv('twitter_training.csv')
validation_df = pd.read_csv('twitter_validation.csv')

print(train_df.columns)



train_df.iloc[:, 2].fillna(-1, inplace=True)
validation_df['Sentiment'].fillna(-1, inplace=True)


train_df.iloc[:, 3].fillna('', inplace=True)

validation_df['Tweet'].fillna('', inplace=True)


tfidf_vectorizer = TfidfVectorizer()


X_train = tfidf_vectorizer.fit_transform(train_df['Tweet'])
y_train = train_df['Sentiment']


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)


X_validation = tfidf_vectorizer.transform(validation_df['Tweet'])
y_validation = validation_df['Sentiment']


y_pred = svm_classifier.predict(X_validation)


print("Classification Report:")
print(classification_report(y_validation, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_validation, y_pred))


