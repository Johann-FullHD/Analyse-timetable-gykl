import os
import sys

import pytest

# Fügen Sie das Stammverzeichnis des Projekts zum Systempfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Tests für die Anwendung
def test_app_modules_exist():
    """Testet, ob die wichtigsten Module existieren."""
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'app.py'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'visualization_utils.py'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'data_utils.py'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'start.py'))

def test_requirements_files_exist():
    """Testet, ob die Requirements-Dateien existieren."""
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'requirements.txt'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'requirements-core.txt'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'requirements-optional.txt'))

def test_data_directories_exist():
    """Testet, ob die Datenverzeichnisse existieren."""
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'csv'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'json'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'svg'))

def test_documentation_files_exist():
    """Testet, ob die Dokumentationsdateien existieren."""
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'README.md'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'LICENSE'))
    assert os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'CONTRIBUTING.md'))

def test_python_version():
    """Testet, ob die Python-Version kompatibel ist."""
    assert sys.version_info >= (3, 8), "Python 3.8 oder höher wird benötigt"

# Haupttest-Ausführung
if __name__ == "__main__":
    pytest.main(["-v"])
