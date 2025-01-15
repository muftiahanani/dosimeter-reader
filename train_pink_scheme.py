from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import pickle

# Baca dataset
def train_pink_scheme():
    print("Memulai pelatihan model Pink Scheme...")
    
    # Membaca dataset
    dataset = pd.read_csv('dataset_dosimeter.csv')
    features = ['Red', 'Green', 'Blue', 'Hue', 'Saturation', 'Value']
    y = dataset['Dose']

    # Analisis performa regresi untuk setiap fitur
    significant_features = []
    for feature in features:
        X_single = dataset[[feature]]
        X_train, X_test, y_train, y_test = train_test_split(X_single, y, test_size=0.2, random_state=42)
        
        # Latih model dengan satu fitur
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"Fitur: {feature}, R²: {r2:.2f}")
        
        # Pilih fitur dengan R² > 0.95
        if r2 > 0.95:
            significant_features.append(feature)

    print(f"Fitur signifikan berdasarkan R² > 0.95: {significant_features}")

    # Latih model dengan fitur signifikan
    X_selected = dataset[significant_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi model akhir
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Akhir - Mean Squared Error: {mse}")
    print(f"Model Akhir - R²: {r2}")

    # Simpan model dan fitur signifikan ke file
    model_data = {'model': model, 'features': significant_features}
    with open('model_random_forest.pkl', 'wb') as file:
        pickle.dump(model_data, file)
    print("Model dan fitur signifikan berhasil disimpan ke 'model_random_forest.pkl'.")

if __name__ == "__main__":
    train_pink_scheme()
