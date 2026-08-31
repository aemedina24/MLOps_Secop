"""
Tests de la capa de persistencia RAW de la ingesta de SECOP II.
"""

import pyarrow as pa
import pyarrow.parquet as pq
from mlops_secop.data.ingest_secop import COLUMNS, _save_page_as_parquet


def test_save_page_creates_file_with_all_22_columns(tmp_path):
    page = [{"nivel_entidad": "Nacional", "valor_contrato": "1000000"}]
    file_path = _save_page_as_parquet(page, run_dir=tmp_path, page_number=1)
    table = pq.read_table(file_path)

    assert set(table.column_names) == set(COLUMNS)
    assert table.num_rows == 1


def test_save_page_fills_missing_fields_with_null(tmp_path):
    page = [{"nivel_entidad": "Nacional"}]
    file_path = _save_page_as_parquet(page, run_dir=tmp_path, page_number=1)
    table = pq.read_table(file_path)
    row = table.to_pylist()[0]

    assert row["nivel_entidad"] == "Nacional"
    assert row["valor_contrato"] is None


def test_save_page_all_columns_are_string_type(tmp_path):
    page = [{"valor_contrato": "5000000"}]
    file_path = _save_page_as_parquet(page, run_dir=tmp_path, page_number=1)
    table = pq.read_table(file_path)

    for field in table.schema:
        assert field.type == pa.string()
