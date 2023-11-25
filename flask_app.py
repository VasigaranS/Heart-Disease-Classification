# flask_app.py
from flask import Flask, render_template, jsonify, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve

app = Flask(__name__)

def load_data(file_path):
    return pd.read_csv(file_path)
    print("here")

def preprocess_data(df):
    df.dropna(axis=0,inplace=True)
    df=df.drop(['currentSmoker'],axis=1)
    X = df.drop('TenYearCHD', axis=1)
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler=StandardScaler()

    X_train_scaled=scaler.fit_transform(X_train)
    X_train=pd.DataFrame(X_train_scaled)

    X_test_scaled=scaler.transform(X_test)
    X_test=pd.DataFrame(X_test_scaled)
    return X_train,y_train,X_test,y_test

def train_model(X_train, y_train):
    model = LogisticRegression(C= 100, class_weight= 'balanced', penalty= 'l2', solver= 'liblinear')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    cm=confusion_matrix(y_test,y_pred)
    TN = cm[0,0]
    TP = cm[1,1]
    FN = cm[1,0]
    FP = cm[0,1]
    sensitivity = TP/float(TP+FN)
    specificity = TN/float(TN+FP)
    y_pred_prob = model.predict_proba(X_test)[:,:]
    auc=roc_auc_score(y_test,y_pred_prob[:,1])

    return auc, sensitivity, accuracy ,classification_rep

# Load and preprocess data

data = load_data('framingham.csv')
X_train,y_train,X_test,y_test = preprocess_data(data)
# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
auc, sensitivity, accuracy ,classification_rep = evaluate_model(model, X_test, y_test)

# API endpoint for model predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()

        # Make predictions
        input_features = [input_data[attribute] for attribute in X.columns]
        prediction = model.predict([input_features])[0]
        confidence = model.predict_proba([input_features])[0][prediction]

        return jsonify({
            'predicted_class': prediction,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# API endpoint for initial analysis results
@app.route('/api/results')
def get_results():
    return jsonify({
        'accuracy': accuracy,
        'classification_rep': classification_rep,
        'auc': auc,
        'sensitivity':sensitivity
    })

if __name__ == '__main__':
    app.run(debug=True)
