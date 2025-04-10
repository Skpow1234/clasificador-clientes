from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load('adaboost_model_cc.pkl')
scaler = joblib.load('scaler.pkl')

# Columnas esperadas (en mayúsculas)
FEATURE_NAMES = [
    'BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
    'PAYMENTS', 'MINIMUM_PAYMENTS', 'PAYMENT_RATIO'
]

# Etiquetas de los clusters
cluster_labels = {
    0: "💼 Cliente conservador",
    1: "🔄 Cliente activo con riesgo",
    2: "💳 Cliente premium o solvente"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    # Valores por defecto
    form_data = {
        'BALANCE': 1500.0,
        'PURCHASES': 500.0,
        'CASH_ADVANCE': 0.0,
        'CREDIT_LIMIT': 2000.0,
        'PAYMENTS': 800.0,
        'MINIMUM_PAYMENTS': 300.0,
        'PAYMENT_RATIO': 0.15
    }

    if request.method == 'POST':
        try:
            # Leer y limpiar valores del formulario
            for field in FEATURE_NAMES:
                raw_value = request.form.get(field, "")
                cleaned = raw_value.replace(",", ".")
                form_data[field] = float(cleaned)

            # Convertir a DataFrame y predecir
            input_df = pd.DataFrame([form_data], columns=FEATURE_NAMES)
            features_scaled = scaler.transform(input_df)
            cluster = model.predict(features_scaled)[0]
            prediction = cluster_labels.get(cluster, f"Cluster {cluster}")

        except Exception as e:
            prediction = f'Error: {str(e)}'
            print(f"⚠️ Error en la predicción: {e}")

    return render_template('index.html', prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
