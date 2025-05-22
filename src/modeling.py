from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

def prepare_model_data(df):
    df['IS_ACCIDENT_ZONE'] = df['INJURIES_TOTAL'].apply(lambda x: 1 if x > 0 else 0)
    y = df['IS_ACCIDENT_ZONE']
    X = df.drop(columns=['IS_ACCIDENT_ZONE', 'CRASH_RECORD_ID', 'CRASH_DATE', 'CRASH_DATE_EST_I',
                         'DATE_POLICE_NOTIFIED', 'LOCATION', 'STREET_NAME'], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def train_models(X_train, X_test, y_train, y_test):
    models = {}
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = (lr, lr.predict(X_test))

    rf = RandomForestClassifier(n_estimators=30, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = (rf, rf.predict(X_test))

    return models
