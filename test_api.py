from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

MALIGNO = {"worst_perimeter": 138.1, "worst_radius": 21.08,
           "worst_area": 1349.0, "worst_concave_points": 0.1974}
BENIGNO = {"worst_perimeter": 93.63, "worst_radius": 14.45,
           "worst_area": 624.1, "worst_concave_points": 0.08045}


def test_inicio():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"estado": "API funcionando"}


def test_modelo():
    r = client.get("/modelo")
    assert r.status_code == 200
    assert r.json()["threshold"] == 4


def test_predict_maligno():
    r = client.post("/predict", json=MALIGNO)
    assert r.status_code == 200
    assert r.json()["prediccion"] == "Maligno"
    assert r.json()["suma"] == 2


def test_predict_benigno():
    r = client.post("/predict", json=BENIGNO)
    assert r.status_code == 200
    assert r.json()["prediccion"] == "Benigno"
    assert r.json()["suma"] == 4


def test_validacion_falta_dato():
    assert client.post("/predict", json={"worst_radius": 15.0}).status_code == 422


def test_validacion_valor_negativo():
    malo = {**BENIGNO, "worst_area": -5}
    assert client.post("/predict", json=malo).status_code == 422


def test_batch():
    r = client.post("/predict/batch", json={"pacientes": [MALIGNO, BENIGNO]})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert [x["prediccion"] for x in data["resultados"]] == ["Maligno", "Benigno"]
