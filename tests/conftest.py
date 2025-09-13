# Datei für Testconfigurations und gemeinsame Fixtures
import os
import sys

import pytest

# Fügen Sie das Stammverzeichnis des Projekts zum Systempfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Konfiguration für alle Tests
def pytest_configure(config):
    """Konfiguriert die Testumgebung."""
    # Markierungen für Tests definieren
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "visualization: marks tests for visualization functions")
    config.addinivalue_line("markers", "data: marks tests for data processing functions")

# Gemeinsame Test-Fixtures für alle Tests
@pytest.fixture(scope="session")
def test_data_path():
    """Liefert den Pfad zum Testdatenverzeichnis."""
    return os.path.join(os.path.dirname(__file__), 'test_data')

@pytest.fixture(scope="session")
def ensure_test_data_dir(test_data_path):
    """Stellt sicher, dass das Testdatenverzeichnis existiert."""
    os.makedirs(test_data_path, exist_ok=True)
    return test_data_path
