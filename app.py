"""
API de cotización inteligente - Forma3D
Expone un endpoint /predecir-precio que recibe las 6 características
(3 elegidas por el cliente + 3 extraídas del STL) y devuelve el precio estimado.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Forma3D - API de Cotización Inteligente",
    description="Predice el precio de una impresión 3D a partir de parámetros de impresión y características geométricas extraídas del STL.",
    version="1.0.0",
)

# CORS: permite que el plugin de WordPress (dominio distinto) llame a esta API desde el navegador del cliente.
# En producción, reemplaza "*" por el dominio real de tu WordPress, ej: ["https://forma3d.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Cargar el modelo entrenado (una sola vez, al iniciar la API)
# ------------------------------------------------------------------
try:
    artefacto = joblib.load("modelo_precio_3d.joblib")
    MODELO = artefacto["modelo"]
    FEATURE_COLUMNS = artefacto["feature_columns"]
    MATERIALES_VALIDOS = artefacto["materiales_validos"]
except FileNotFoundError:
    raise RuntimeError(
        "No se encontró modelo_precio_3d.joblib. Ejecuta entrenar_modelo.py primero."
    )


# ------------------------------------------------------------------
# Esquema de entrada (lo que el plugin de WordPress debe enviar)
# ------------------------------------------------------------------
class SolicitudCotizacion(BaseModel):
    material: str = Field(..., description="PLA, ABS o TPU", examples=["PLA"])
    altura_capa: float = Field(..., gt=0, description="mm, ej. 0.1 / 0.2 / 0.3", examples=[0.2])
    porcentaje_relleno: float = Field(..., ge=0, le=100, description="% infill", examples=[20])
    volumen_mm3: float = Field(..., gt=0, description="Extraído del STL", examples=[45000])
    area_superficial_mm2: float = Field(..., gt=0, description="Extraído del STL", examples=[8000])
    num_triangulos: int = Field(..., gt=0, description="Extraído del STL", examples=[5000])


class RespuestaCotizacion(BaseModel):
    precio_estimado: float
    moneda: str = "USD"


# ------------------------------------------------------------------
# Endpoint principal
# ------------------------------------------------------------------
@app.post("/predecir-precio", response_model=RespuestaCotizacion)
def predecir_precio(solicitud: SolicitudCotizacion):
    material = solicitud.material.upper().strip()

    if material not in MATERIALES_VALIDOS:
        raise HTTPException(
            status_code=400,
            detail=f"Material '{solicitud.material}' no válido. Debe ser uno de: {MATERIALES_VALIDOS}",
        )

    # Construir el vector de entrada con el mismo formato (dummy encoding) usado en entrenamiento
    fila = {col: 0 for col in FEATURE_COLUMNS}
    fila["Altura_Capa"] = solicitud.altura_capa
    fila["Porcentaje_Relleno"] = solicitud.porcentaje_relleno
    fila["Volumen_mm3"] = solicitud.volumen_mm3
    fila["Area_Superficial_mm2"] = solicitud.area_superficial_mm2
    fila["Num_Triangulos"] = solicitud.num_triangulos
    fila[f"Material_{material}"] = 1

    entrada = pd.DataFrame([fila])[FEATURE_COLUMNS]  # mismo orden de columnas que en entrenamiento

    precio_predicho = MODELO.predict(entrada)[0]
    precio_predicho = max(round(float(precio_predicho), 2), 0.0)

    return RespuestaCotizacion(precio_estimado=precio_predicho)


@app.get("/")
def estado():
    return {"status": "ok", "servicio": "Forma3D API de cotización"}
