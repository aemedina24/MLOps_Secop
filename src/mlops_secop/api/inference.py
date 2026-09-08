"""
Servicio de inferencia: carga el modelo/encoder/tablas de referencia
UNA sola vez, y transforma un contrato nuevo con la MISMA logica que
build_features.py + train_isolation_forest.prepare_matrix -- para que
no haya "train/serve skew" (que el modelo vea en produccion features
calculadas de forma distinta a como se calcularon en entrenamiento).

Decision de diseno importante: `mes_firma` se calcula con `.month` de
Python (equivalente exacto de MONTH() en DuckDB, sin ambiguedad), pero
`dia_semana_firma` se calcula pidiendole el numero a DuckDB
(`DAYOFWEEK(...)`) en vez de reimplementarlo con `.weekday()` o
`.isoweekday()` de Python -- esas dos funciones usan una convencion de
numeracion DISTINTA a la de DuckDB (Python: lunes=0 o lunes=1; DuckDB:
domingo=0), y reimplementar mal esa conversion introduciria un sesgo
silencioso entre el dato de entrenamiento y el de inferencia. DuckDB ya
es una dependencia del proyecto, así que reutilizarlo aqui es mas
seguro que traducir la convencion a mano.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from scipy import sparse

from mlops_secop.api.schemas import ContratoInput, PrediccionOutput
from mlops_secop.model.build_reference_tables import (
    _CATEGORIA_COLS_FEATURES,
    CATEGORIA_PROMEDIOS_PATH,
    PROVEEDOR_CONTEOS_PATH,
)
from mlops_secop.model.train_isolation_forest import (
    CATEGORICAL_COLS,
    ENCODER_PATH,
    MODEL_PATH,
    NUMERIC_COLS,
)

logger = logging.getLogger(__name__)


class ModeloNoDisponibleError(RuntimeError):
    """El modelo o alguno de sus artefactos de referencia no existe en disco."""


def _dia_semana_duckdb(fecha) -> int:
    """DAYOFWEEK de DuckDB: domingo=0 ... sabado=6 (misma convencion
    usada al construir features.parquet)."""
    con = duckdb.connect()
    try:
        return con.execute(
            "SELECT DAYOFWEEK(CAST(? AS DATE))", [fecha.isoformat()]
        ).fetchone()[0]
    finally:
        con.close()


class InferenceService:
    def __init__(
        self,
        model_path: Path | None = None,
        encoder_path: Path | None = None,
        categoria_promedios_path: Path | None = None,
        proveedor_conteos_path: Path | None = None,
    ) -> None:
        # No usar las constantes del modulo como VALOR POR DEFECTO del
        # parametro -- eso las "congelaria" en el momento en que Python
        # importa este archivo, y romperia monkeypatch.setattr(...) en
        # los tests (mismo problema que config.py ya documenta y evita
        # con funciones en vez de constantes). Referenciarlas aqui,
        # DENTRO del cuerpo del metodo, si permite que los tests las
        # sobreescriban.
        model_path = model_path or MODEL_PATH
        encoder_path = encoder_path or ENCODER_PATH
        categoria_promedios_path = categoria_promedios_path or CATEGORIA_PROMEDIOS_PATH
        proveedor_conteos_path = proveedor_conteos_path or PROVEEDOR_CONTEOS_PATH

        for path in (
            model_path,
            encoder_path,
            categoria_promedios_path,
            proveedor_conteos_path,
        ):
            if not path.exists():
                raise ModeloNoDisponibleError(
                    f"Artefacto requerido no encontrado: {path}. Corre "
                    "'uv run python -m mlops_secop.model.train_isolation_forest' "
                    "y 'uv run python -m mlops_secop.model.build_reference_tables' "
                    "antes de levantar la API."
                )

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

        promedios_df = pd.read_parquet(categoria_promedios_path)
        col_a, col_b = _CATEGORIA_COLS_FEATURES
        self._promedio_categoria: dict[tuple[str, str], float] = {
            (getattr(row, col_a), getattr(row, col_b)): row.promedio_categoria
            for row in promedios_df.itertuples(index=False)
        }
        self._promedio_global = float(promedios_df["promedio_categoria"].mean())

        conteos_df = pd.read_parquet(proveedor_conteos_path)
        self._concentracion_proveedor: dict[str, int] = dict(
            zip(
                conteos_df["documento_proveedor"],
                conteos_df["concentracion_proveedor"],
            )
        )

        logger.info(
            "InferenceService listo: %s combinaciones categoria, %s "
            "proveedores conocidos.",
            len(self._promedio_categoria),
            len(self._concentracion_proveedor),
        )

    def _promedio_para(self, modalidad: str, departamento: str) -> float:
        clave = (modalidad, departamento)
        if clave in self._promedio_categoria:
            return self._promedio_categoria[clave]
        logger.warning(
            "Combinacion modalidad/departamento no vista en el historico "
            "(%s, %s); usando promedio global como fallback.",
            modalidad,
            departamento,
        )
        return self._promedio_global

    def _concentracion_para(self, documento_proveedor: str) -> int:
        if documento_proveedor in self._concentracion_proveedor:
            # +1: este contrato nuevo se sumaria al historico del proveedor.
            return int(self._concentracion_proveedor[documento_proveedor]) + 1
        logger.warning(
            "Proveedor %s no visto en el historico; asumiendo "
            "concentracion_proveedor=1 (su primer contrato conocido).",
            documento_proveedor,
        )
        return 1

    def _construir_fila(self, contrato: ContratoInput) -> pd.DataFrame:
        duracion_dias = (
            contrato.fecha_fin_ejecucion - contrato.fecha_inicio_ejecucion
        ).days
        promedio_categoria = self._promedio_para(
            contrato.modalidad_de_contratacion, contrato.departamento_entidad
        )
        valor_vs_promedio = (
            contrato.valor_contrato / promedio_categoria if promedio_categoria else 0.0
        )

        fila = {
            "valor_contrato": contrato.valor_contrato,
            "valor_vs_promedio_categoria": valor_vs_promedio,
            "duracion_dias": duracion_dias,
            "mes_firma": contrato.fecha_de_firma_del_contrato.month,
            "dia_semana_firma": _dia_semana_duckdb(
                contrato.fecha_de_firma_del_contrato
            ),
            "concentracion_proveedor": self._concentracion_para(
                contrato.documento_proveedor
            ),
            "modalidad_de_contratacion": contrato.modalidad_de_contratacion,
            "departamento_entidad": contrato.departamento_entidad,
            "nivel_entidad": contrato.nivel_entidad,
            "tipo_de_contrato": contrato.tipo_de_contrato,
            "origen": contrato.origen,
            "tipo_documento_proveedor": contrato.tipo_documento_proveedor,
        }
        return pd.DataFrame([fila])

    def predict(self, contrato: ContratoInput) -> PrediccionOutput:
        df = self._construir_fila(contrato)
        numeric_matrix = sparse.csr_matrix(df[NUMERIC_COLS].to_numpy(dtype=float))
        categorical_matrix = self.encoder.transform(df[CATEGORICAL_COLS])
        X = sparse.hstack([numeric_matrix, categorical_matrix], format="csr")

        raw_score = self.model.decision_function(X)[0]
        is_anomaly = bool(self.model.predict(X)[0] == -1)

        return PrediccionOutput(
            numero_del_contrato=contrato.numero_del_contrato,
            anomaly_score=float(-raw_score),
            is_anomaly=is_anomaly,
        )


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """
    Singleton reutilizado entre requests -- cargar el modelo/encoder/
    tablas de referencia en cada request seria lento e innecesario, ya
    que no cambian mientras la API esta corriendo.
    """
    return InferenceService()
