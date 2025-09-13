# Projektstruktur der Kursplan-Analyse

Diese Datei beschreibt die Ordnerstruktur und Organisation des Kursplan-Analyse-Projekts.

## Hauptverzeichnisse

- `csv/` - CSV-Beispieldaten für Kurse, Teilnehmer und Stundenpläne
- `json/` - JSON-Datenformate für Kurse, Teilnehmer und Stundenpläne
- `svg/` - SVG-Exportbeispiele und Visualisierungen
- `logs/` - Logdateien (wird automatisch erstellt)
- `data/` - Lokale Datenbankdateien (wird automatisch erstellt)
- `tests/` - Testdateien für die Anwendung (zukünftige Erweiterung)

## Hauptdateien

- `start.py` - Einstiegspunkt und Startup-Manager für die Anwendung
- `app.py` - Hauptanwendung mit Streamlit-Interface
- `visualization_utils.py` - Funktionen für Datenvisualisierung
- `data_utils.py` - Funktionen für Datenverarbeitung
- `requirements.txt` - Vollständige Liste aller Abhängigkeiten
- `requirements-core.txt` - Minimale Abhängigkeiten für Grundfunktionalität
- `requirements-optional.txt` - Optionale Abhängigkeiten für erweiterte Funktionen
- `config.toml` - Allgemeine Konfigurationseinstellungen
- `README.md` - Hauptdokumentation
- `LICENSE` - Lizenzinformationen
- `CONTRIBUTING.md` - Anleitung zum Beitragen zum Projekt
- `SAMPLE_DATA.md` - Beschreibung der Beispieldaten

## Erweiterbare Struktur

Das Projekt ist modular aufgebaut und kann leicht erweitert werden:

- Neue Visualisierungsfunktionen können in `visualization_utils.py` hinzugefügt werden
- Neue Datenverarbeitungsfunktionen können in `data_utils.py` hinzugefügt werden
- Das UI kann in `app.py` erweitert werden

## Zukünftige Entwicklung

Geplante Erweiterungen für die Projektstruktur:

- `tests/` - Einheitstests und Integrationstests
- `docs/` - Erweiterte Dokumentation
- `api/` - RESTful API für die Datenverarbeitung
- `plugins/` - Plugin-Schnittstelle für Erweiterungen
