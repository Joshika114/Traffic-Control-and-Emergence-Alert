from src.preprocessing import load_dataset, clean_crash_data
from src.modeling import prepare_model_data, train_models
from src.evaluation import evaluate_model

def run_pipeline():
    df = load_dataset("data/Traffic_Crashes_-_Crashes.csv")
    df_cleaned = clean_crash_data(df)
    X_train, X_test, y_train, y_test = prepare_model_data(df_cleaned)
    models = train_models(X_train, X_test, y_train, y_test)

    for name, (model, y_pred) in models.items():
        evaluate_model(name, y_test, y_pred)

if __name__ == "__main__":
    run_pipeline()
