from mlops_secop import hello


def test_hello_returns_expected_string():
    """Verifica que hello() retorna el saludo esperado del paquete."""
    result = hello()
    assert result == "Hello from mlops-secop!"


def test_hello_returns_string_type():
    """Verifica que hello() siempre retorna un str, no None u otro tipo."""
    result = hello()
    assert isinstance(result, str)
