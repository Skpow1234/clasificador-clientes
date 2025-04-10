# 💳 Clasificador de Clientes con Flask + AdaBoost + Docker

Este proyecto es una aplicación web desarrollada con **Flask** que permite clasificar a los clientes de tarjetas de crédito en grupos según su comportamiento financiero. Se basa en un modelo de **clustering con KMeans** y un modelo de **clasificación con AdaBoost**, entrenado para predecir el grupo al que pertenece un nuevo cliente. El modelo se ha guardado como un archivo `.pkl` para facilitar su despliegue y reutilización.

## 🔗 Repositorio de Google Colab con el modelo

- [Ver notebook con el modelo de clustering y clasificación](https://colab.research.google.com/drive/1C5QgTnw8_I86KJQoWqynF0Q33NWNjiyd?usp=sharing)

---

## 📦 Características del proyecto

- Clasificación supervisada con **AdaBoostClassifier**
- Agrupamiento no supervisado con **KMeans** (`k=3`)
- Interfaz web construida con **Flask + Bootstrap**
- Formulario con valores predefinidos para pruebas
- Estilo visual en modo oscuro
- Modelo y scaler guardados en `.pkl`
- Despliegue rápido con **Docker + Gunicorn**

---

## 📊 Variables utilizadas

El modelo usa solo **7 variables clave** para representar el comportamiento del cliente:

| Variable             | Descripción                                      |
|----------------------|--------------------------------------------------|
| `BALANCE`            | Saldo promedio de la cuenta                      |
| `PURCHASES`          | Monto total de compras                           |
| `CASH_ADVANCE`       | Anticipos de efectivo realizados                 |
| `CREDIT_LIMIT`       | Límite de crédito asignado                       |
| `PAYMENTS`           | Total de pagos realizados                        |
| `MINIMUM_PAYMENTS`   | Pagos mínimos efectuados                         |
| `PAYMENT_RATIO`      | Relación entre pago mínimo y límite de crédito   |

---

## Cómo ejecutar el proyecto

### 1. Clona el repositorio

```bash
git clone https://github.com/Skpow1234/clasificador-clientes.git
cd clasificador-clientes
```

### 2. Ejecuta con Docker

docker-compose up --build

La aplicación estará disponible en:
📍 <http://localhost:5007>
