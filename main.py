import json
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------
# Logs: registran cada predicción para monitorear el uso del modelo
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("predicciones.log", encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("api-cancer-mp")

# ---------------------------------------------------------------
# Cargar el modelo entrenado (artefacto generado en el notebook)
# ---------------------------------------------------------------
RUTA_MODELO = Path(__file__).parent / "modelo_mp.json"
with open(RUTA_MODELO, "r", encoding="utf-8") as f:
    MODELO = json.load(f)

FEATURES = MODELO["features"]
CORTES = MODELO["cortes"]
THRESHOLD = MODELO["threshold"]

app = FastAPI(
    title="API Detección de Cáncer de Mama - Neurona MP",
    description="Ejercicio académico. NO reemplaza un diagnóstico médico.",
    version="1.0.0",
)


class Paciente(BaseModel):
    worst_perimeter: float = Field(..., gt=0, description="Peor perímetro")
    worst_radius: float = Field(..., gt=0, description="Peor radio")
    worst_area: float = Field(..., gt=0, description="Peor área")
    worst_concave_points: float = Field(..., ge=0, description="Peor cantidad de puntos cóncavos")


class LotePacientes(BaseModel):
    pacientes: List[Paciente] = Field(..., min_length=1, max_length=1000)


def binarizar(nombre: str, valor: float) -> int:
    """Valor <= punto de corte -> 1, si no -> 0 (mismo criterio del entrenamiento)."""
    return 1 if valor <= CORTES[nombre] else 0


def clasificar(paciente: Paciente) -> dict:
    datos = paciente.model_dump()
    entradas_bin = {f: binarizar(f, datos[f.replace(" ", "_")]) for f in FEATURES}
    suma = sum(entradas_bin.values())
    clase = int(suma >= THRESHOLD)
    return {
        "prediccion": "Benigno" if clase == 1 else "Maligno",
        "clase": clase,
        "entradas_binarias": entradas_bin,
        "suma": suma,
        "threshold": THRESHOLD,
    }


@app.get("/")
def inicio():
    return {"estado": "API funcionando"}


@app.get("/modelo")
def info_modelo():
    return MODELO


@app.post("/predict")
def predecir(paciente: Paciente):
    resultado = clasificar(paciente)
    log.info("predict | entrada=%s | resultado=%s", paciente.model_dump(), resultado["prediccion"])
    return resultado


@app.post("/predict/batch")
def predecir_lote(lote: LotePacientes):
    resultados = [clasificar(p) for p in lote.pacientes]
    log.info("predict/batch | n=%d", len(resultados))
    return {"total": len(resultados), "resultados": resultados}
