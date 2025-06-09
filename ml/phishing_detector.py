import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def main():
    # Load dataset
    emails = pd.read_csv('Cleaned_data.csv')

    # Features and target
    X_text = emails['text']
    y = emails['label_num']  # Already numeric: 0 = ham, 1 = spam

    # Vectorize email text
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(X_text)

    # Split into train/test
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
    threshold = 0.6
    y_pred_custom = (y_probs >= threshold).astype(int)

    # Classification report
    print("Classification Report with threshold 0.6:\n")
    print(classification_report(y_test, y_pred_custom, target_names=["ham", "spam"]))

    # Save model and vectorizer
    with open('ml/phishing_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('ml/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    main()
