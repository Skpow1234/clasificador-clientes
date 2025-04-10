from flask import Flask, render_template, request
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cargar modelo y scaler
try:
    model = joblib.load('adaboost_model_cc.pkl')
    scaler = joblib.load('scaler.pkl')
    logger.info("Modelo y scaler cargados exitosamente")
except Exception as e:
    logger.error(f"Error cargando el modelo o scaler: {str(e)}", exc_info=True)
    raise

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
    error = None
    
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
            logger.info("Procesando solicitud POST")
            # Leer y limpiar valores del formulario
            for field in FEATURE_NAMES:
                raw_value = request.form.get(field, "")
                if not raw_value:
                    raise ValueError(f"El campo {field} es requerido")
                cleaned = raw_value.replace(",", ".")
                try:
                    value = float(cleaned)
                    if value < 0:
                        raise ValueError(f"El campo {field} no puede ser negativo")
                    form_data[field] = value
                except ValueError:
                    raise ValueError(f"El valor en {field} debe ser un número válido")

            # Convertir a DataFrame y predecir
            input_df = pd.DataFrame([form_data], columns=FEATURE_NAMES)
            features_scaled = scaler.transform(input_df)
            cluster = model.predict(features_scaled)[0]
            prediction = cluster_labels.get(cluster, f"Cluster {cluster}")
            logger.info(f"Predicción exitosa: {prediction}")

        except Exception as e:
            error = f'Error: {str(e)}'
            logger.error(f"Error en la predicción: {str(e)}", exc_info=True)

    return render_template('index.html', prediction=prediction, error=error, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
