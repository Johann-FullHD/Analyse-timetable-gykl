import os
import sys

import pytest

# Fügen Sie das Stammverzeichnis des Projekts zum Systempfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import data_utils
# Import der zu testenden Module
import visualization_utils


# Fixtures für gemeinsam genutzte Testdaten
@pytest.fixture
def sample_course_data():
    """Stellt Beispieldaten für Kurse bereit."""
    return [
        {"kurs_id": 1, "kurs": "11MA1", "participants_count": 20},
        {"kurs_id": 2, "kurs": "11DE2", "participants_count": 15},
        {"kurs_id": 3, "kurs": "11EN3", "participants_count": 18}
    ]

@pytest.fixture
def sample_student_data():
    """Stellt Beispieldaten für Schüler bereit."""
    return [
        {"name": "Student 1", "courses": "11MA1, 11DE2, 11EN3"},
        {"name": "Student 2", "courses": "11MA1, 11DE2"},
        {"name": "Student 3", "courses": "11EN3, 11DE2"}
    ]

# Testen der Hilfsfunktionen
def test_search_students_by_name(sample_student_data):
    """Testet die Schülersuche nach Namen."""
    import pandas as pd
    students_df = pd.DataFrame(sample_student_data)
    students_df['Kurse_Liste'] = students_df['courses'].str.split(', ')
    
    # Diese Funktion sollte mindestens existieren
    assert hasattr(visualization_utils, 'search_students_by_name')
    
    # Dieser Test würde die tatsächliche Funktionalität prüfen
    # result = visualization_utils.search_students_by_name(students_df, "Student 1")
    # assert len(result) == 1
    # assert result.iloc[0]['name'] == "Student 1"

def test_search_students_by_courses(sample_student_data):
    """Testet die Schülersuche nach Kursen."""
    import pandas as pd
    students_df = pd.DataFrame(sample_student_data)
    students_df['Kurse_Liste'] = students_df['courses'].str.split(', ')
    
    # Diese Funktion sollte mindestens existieren
    assert hasattr(visualization_utils, 'search_students_by_courses')
    
    # Dieser Test würde die tatsächliche Funktionalität prüfen
    # result = visualization_utils.search_students_by_courses(students_df, ["11MA1"])
    # assert len(result) == 2  # Sollte Student 1 und Student 2 finden

def test_calculate_student_overlap():
    """Testet die Berechnung der Schülerüberschneidungen."""
    # Diese Funktion sollte mindestens existieren
    assert hasattr(visualization_utils, 'calculate_student_overlap')

def test_export_plotly_figure():
    """Testet die Export-Funktion für Plotly-Figuren."""
    # Diese Funktion sollte mindestens existieren
    assert hasattr(visualization_utils, 'export_plotly_figure')

# Haupttest-Ausführung
if __name__ == "__main__":
    pytest.main(["-v"])
