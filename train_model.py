import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from core.data_processor import GlucoseProcessor

def run_training_pipeline():
    processor = GlucoseProcessor(threshold=180)
    
    train_dfs = []
    test_dfs = []

    print("--- Starting Model Training Pipeline ---")

    if not os.path.exists('dataset/'):
        print("Error: 'dataset/' folder not found.")
        return

    for filename in os.listdir('dataset/'):
        if filename.endswith('.xml'):
            path = os.path.join('dataset/', filename)
            print(f"Processing: {filename}")
            
            raw_df = processor.parse_ohio_xml(path)
            processed_df = processor.engineer_features(raw_df, is_training=True)
            
            if 'training' in filename.lower():
                train_dfs.append(processed_df)
            elif 'testing' in filename.lower():
                test_dfs.append(processed_df)

    train_data = pd.concat(train_dfs)
    test_data = pd.concat(test_dfs)

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    features = ['glucose', 'slope_15', 'slope_60', 'cob_2h']
    X_train, y_train = train_data[features], train_data['target']
    X_test, y_test = test_data[features], test_data['target']

    print(f"Training on {len(X_train)} samples...")

    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    print("\n--- Training Results ---")
    print(f"Model AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))

    os.makedirs('models', exist_ok=True)
    joblib.dump({
        'model': model,
        'features': features
    }, 'models/spike_rf.joblib')
    
    print("\nModel saved successfully as 'models/spike_rf.joblib'")

if __name__ == "__main__":
    run_training_pipeline()