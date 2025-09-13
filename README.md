# Kursplan-Analyse 📊

Eine leistungsstarke Anwendung zur Analyse, Visualisierung und Optimierung von Kursplänen und Stundenplanbelegungen in Bildungseinrichtungen.

## 🌟 Features

- **Datenvisualisierung**: Interaktive Heatmaps, Netzwerkdiagramme und 3D-Visualisierungen
- **Stundenplananalyse**: Erkennung von Überschneidungen und Optimierungsmöglichkeiten
- **Schüler/Kurs-Beziehungen**: Analyse von Kursbelegungen und gemeinsamen Kursen
- **Exportfunktionen**: SVG, PNG, Excel und PDF-Exports für alle Visualisierungen
- **Performance-Optimierung**: Caching, Multi-Threading und optimierte Datenverarbeitung

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/[ihr-username]/kursplan-analyse.git
cd kursplan-analyse

# Abhängigkeiten installieren
pip install -r requirements.txt

# Anwendung starten
python start.py
```

### Minimale Installation

Für eine minimale Installation mit nur den wesentlichen Funktionen:

```bash
pip install -r requirements-core.txt
```

## 🖥️ Verwendung

1. Starten Sie die Anwendung mit `python start.py`
2. Laden Sie Ihre Kursdaten im CSV, JSON oder Excel-Format hoch
3. Nutzen Sie die verschiedenen Analyse- und Visualisierungstools
4. Exportieren Sie die Ergebnisse in verschiedenen Formaten

## 📋 Datenformate

Die Anwendung unterstützt folgende Eingabeformate:

- **CSV**: Komma- oder Semikolon-getrennte Dateien
- **JSON**: Strukturierte Datensätze im JSON-Format
- **Excel**: XLSX-Dateien mit Tabellenblättern für Kurse und Teilnehmer

## 🔧 Erweiterte Konfiguration

Die `start.py` unterstützt verschiedene Kommandozeilenparameter:

```
python start.py [--debug] [--port PORT] [--browser BROWSER] [--theme THEME] [--turbo]
```

| Parameter | Beschreibung |
|-----------|--------------|
| `--debug` | Aktiviert den Debug-Modus |
| `--port`  | Port für den Streamlit-Server (Standard: 8501) |
| `--browser` | Browser-Steuerung ('new', 'none') |
| `--theme` | Design-Theme ('modern', 'classic', 'dark') |
| `--turbo` | Aktiviert Leistungsoptimierungen |

## 🤝 Beiträge

Beiträge sind willkommen! Bitte lesen Sie [CONTRIBUTING.md](CONTRIBUTING.md) für Details zum Prozess für Pull Requests.

## 📄 Lizenz

Dieses Projekt steht unter der MIT-Lizenz - siehe die [LICENSE](LICENSE) Datei für Details.

## 📞 Kontakt

Bei Fragen oder Anregungen erstellen Sie bitte ein Issue im GitHub-Repository.
