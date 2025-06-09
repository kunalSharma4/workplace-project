import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

def main():
    # 1. Load cleaned dataset
    emails = pd.read_csv('Cleaned_data.csv')

    # 2. Select features and labels
    X_text = emails['Email Text']       # Text of emails (input features)
    y_labels = emails['Email Type']     # Labels (Safe or Phishing)

    # 3. Convert text data into numeric features using TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7,min_df=5, 
    ngram_range=(1,2), 
    max_features=5000)
    X = vectorizer.fit_transform(X_text)

    # 4. Encode text labels into numbers (e.g., Phishing=0, Safe=1)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # 5. Split dataset into training set and testing set (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Train a Logistic Regression model
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)

    # 7. Predict on test data and evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 8. Save the trained model and vectorizer for later use
    with open('ml/phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('ml/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    main()


















import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import numpy as np

def main():
    emails = pd.read_csv('Cleaned_data.csv')
    X_text = emails['Email Text']
    y_labels = emails['Email Type']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(X_text)

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_macro')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")

    model = grid_search.best_estimator_

    # Predict with custom threshold
    y_probs = model.predict_proba(X_test)[:, 1]
    threshold = 0.6  # You can try tuning this value
    y_pred_custom = (y_probs >= threshold).astype(int)

    print("Classification Report with threshold 0.6:\n")
    print(classification_report(y_test, y_pred_custom, target_names=le.classes_))

    with open('ml/phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('ml/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    main()
