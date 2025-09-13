# Mitwirken am Kursplan-Analyse-Projekt

Vielen Dank für Ihr Interesse an diesem Projekt! Wir freuen uns über jede Unterstützung zur Verbesserung der Kursplan-Analyse-Software.

## Wie kann ich beitragen?

Es gibt verschiedene Möglichkeiten, zum Projekt beizutragen:

1. **Code-Beiträge**: Neue Features, Bugfixes oder Performance-Optimierungen
2. **Dokumentation**: Verbesserung der Dokumentation oder Erstellung von Tutorials
3. **Testing**: Testen der Software und Melden von Bugs
4. **Ideen**: Vorschläge für neue Features oder Verbesserungen

## Entwicklungsprozess

1. **Fork** des Repositories auf GitHub
2. Erstellen eines **Feature-Branches** (`git checkout -b feature/amazing-feature`)
3. **Committen** Ihrer Änderungen (`git commit -m 'Add some amazing feature'`)
4. **Push** Ihres Branches (`git push origin feature/amazing-feature`)
5. Eröffnen eines **Pull Requests**

## Pull Request Guidelines

- Stellen Sie sicher, dass Ihr Code den Stilrichtlinien folgt (PEP 8 für Python)
- Fügen Sie Tests für neue Funktionen hinzu
- Aktualisieren Sie die Dokumentation entsprechend
- Beschreiben Sie Ihre Änderungen ausführlich im Pull Request

## Entwicklungsumgebung einrichten

```bash
# Repository klonen
git clone https://github.com/[ihr-username]/kursplan-analyse.git
cd kursplan-analyse

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Entwicklungsabhängigkeiten
```

## Codekonventionen

- Verwenden Sie aussagekräftige Variablen- und Funktionsnamen
- Dokumentieren Sie Ihren Code mit Docstrings im Google-Stil
- Fügen Sie Typhinweise hinzu
- Formatieren Sie Ihren Code mit Black
- Verwenden Sie Pytest für Tests

## Testen

```bash
# Alle Tests ausführen
pytest

# Testabdeckung prüfen
pytest --cov=.
```

## Commit-Nachrichten

- Verwenden Sie klare, beschreibende Commit-Nachrichten
- Beginnen Sie mit einem Verb im Imperativ (z.B. "Add", "Fix", "Update")
- Halten Sie die erste Zeile unter 50 Zeichen
- Fügen Sie bei Bedarf zusätzliche Details in nachfolgenden Zeilen hinzu

## Versionierung

Wir folgen [Semantic Versioning](https://semver.org/):

- MAJOR-Version bei inkompatiblen API-Änderungen
- MINOR-Version bei rückwärtskompatiblen Funktionserweiterungen
- PATCH-Version bei rückwärtskompatiblen Bugfixes

## Fragen?

Bei Fragen oder Unklarheiten erstellen Sie bitte ein Issue im GitHub-Repository.
