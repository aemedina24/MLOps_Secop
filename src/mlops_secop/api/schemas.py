"""
Esquemas de entrada/salida de la API de inferencia (Fase 4).

`ContratoInput` recibe los campos CRUDOS de un contrato -- los que un
usuario real conoce de antemano al registrar un contrato nuevo. NO
incluye `valor_vs_promedio_categoria` ni `concentracion_proveedor`:
esas dos son features derivadas de agregados historicos que la API
calcula internamente (ver `mlops_secop.api.inference`), no datos que
alguien pueda escribir a mano de forma realista.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator


class ContratoInput(BaseModel):
    numero_del_contrato: str = Field(
        ..., description="Identificador del contrato, solo para referencia."
    )
    valor_contrato: float = Field(..., gt=0, description="Valor del contrato en COP.")
    fecha_de_firma_del_contrato: date
    fecha_inicio_ejecucion: date
    fecha_fin_ejecucion: date
    documento_proveedor: str = Field(
        ..., description="Identificador del proveedor/contratista."
    )
    modalidad_de_contratacion: str
    departamento_entidad: str
    nivel_entidad: str
    tipo_de_contrato: str
    origen: str
    tipo_documento_proveedor: str

    @field_validator("fecha_fin_ejecucion")
    @classmethod
    def fin_no_antes_de_inicio(cls, fecha_fin: date, info) -> date:
        fecha_inicio = info.data.get("fecha_inicio_ejecucion")
        if fecha_inicio is not None and fecha_fin < fecha_inicio:
            raise ValueError(
                "fecha_fin_ejecucion no puede ser anterior a fecha_inicio_ejecucion."
            )
        return fecha_fin

    model_config = {
        "json_schema_extra": {
            "example": {
                "numero_del_contrato": "CTO-2026-001",
                "valor_contrato": 45000000,
                "fecha_de_firma_del_contrato": "2026-03-10",
                "fecha_inicio_ejecucion": "2026-03-15",
                "fecha_fin_ejecucion": "2026-09-15",
                "documento_proveedor": "900123456",
                "modalidad_de_contratacion": "Contratación directa",
                "departamento_entidad": "Cundinamarca",
                "nivel_entidad": "Territorial",
                "tipo_de_contrato": "Prestación de servicios",
                "origen": "Nacional",
                "tipo_documento_proveedor": "NIT",
            }
        }
    }


class PrediccionOutput(BaseModel):
    numero_del_contrato: str
    anomaly_score: float = Field(
        ..., description="Score de anomalia; mas alto = mas atipico."
    )
    is_anomaly: bool = Field(
        ..., description="True si el modelo lo marca como contrato atipico."
    )


class HealthOutput(BaseModel):
    status: str
    modelo_cargado: bool
