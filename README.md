# API Detección de Cáncer de Mama – Neurona McCulloch-Pitts + FastAPI

**Estudiante:** Omar Recalde

API que consume el modelo de la neurona MP entrenado con el dataset UCI *Breast Cancer Wisconsin*
(569 muestras, 30 variables). Usa solo las 4 variables con mayor AUC individual:
`worst perimeter`, `worst radius`, `worst area`, `worst concave points`.

> Ejercicio académico. No reemplaza un diagnóstico médico.

## Arquitectura

1. **Entrenamiento (notebook):** dataset → selección Top 4 por AUC → binarización con punto de corte → neurona MP (threshold óptimo = 4) → `modelo_mp.json`.
2. **Servicio (FastAPI):** cliente → validación Pydantic → binarización → neurona MP → respuesta JSON (Benigno / Maligno).

## Estructura

```
api-cancer-mp/
├── API_Arquitectura_Deteccion_Cancer.ipynb    # Parte 1 (entrenamiento)
├── modelo_mp.json                             # Modelo entrenado (artefacto)
├── main.py                                    # API FastAPI
├── test_api.py                                # Pruebas automáticas
├── requirements.txt
├── Dockerfile
└── README.md
```

## Ejecutar localmente

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Verifica que la API funcione |
| GET | `/modelo` | Variables, puntos de corte y threshold |
| POST | `/predict` | Predicción para un paciente |
| POST | `/predict/batch` | Predicción para varios pacientes |

### Ejemplo `/predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"worst_perimeter": 93.63, "worst_radius": 14.45, "worst_area": 624.1, "worst_concave_points": 0.08045}'
```

Respuesta:

```json
{
  "prediccion": "Benigno",
  "clase": 1,
  "entradas_binarias": {"worst perimeter": 1, "worst radius": 1, "worst area": 1, "worst concave points": 1},
  "suma": 4,
  "threshold": 4
}
```

## Pruebas

```bash
pytest -q
```

## Docker

```bash
docker build -t api-cancer-mp .
docker run -p 8000:8000 api-cancer-mp
```

## Logs

Cada predicción se registra en `predicciones.log`.
