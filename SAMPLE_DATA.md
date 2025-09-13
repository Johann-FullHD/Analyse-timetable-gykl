# Beispieldaten für Kursplan-Analyse

In diesem Verzeichnis finden Sie Beispieldaten, die zur Demonstration der Funktionalität der Kursplan-Analyse-Anwendung verwendet werden können.

## Verfügbare Datensätze

### CSV-Dateien

- `kurs_statistik.csv`: Statistische Informationen zu Kursen
- `kurs_teilnehmer.csv`: Zuordnung von Schülern zu Kursen
- `personen_kurse.csv`: Liste der Personen und ihrer belegten Kurse
- `teacher_short.csv`: Kurzinformationen zu Lehrern
- `timetable_11.csv`: Stundenplan für die 11. Jahrgangsstufe

### JSON-Dateien

- `course_participants.json`: Teilnehmerinformationen pro Kurs
- `courses.json`: Detaillierte Kursinformationen
- `students.json`: Schülerdaten mit Kursbelegungen
- `timetable.json`: Vollständiger Stundenplan in JSON-Format

### SVG-Dateien

- Verschiedene SVG-Dateien mit Beispielvisualisierungen

## Datenformat-Spezifikation

### Kurs-Daten (courses.json)

```json
{
  "kurs_id": 123,
  "kurs": "11MA1",
  "kurs_type": "Leistungskurs",
  "participants_count": 25,
  "teacher": "Doe",
  "room": "R101"
}
```

### Schüler-Daten (students.json)

```json
{
  "id": 456,
  "name": "Max Mustermann",
  "courses": "11MA1, 11DE2, 11EN3, ..."
}
```

### Stundenplan (timetable.json)

```json
{
  "tag": "Montag",
  "stunde": 1,
  "faecher": [
    { "fach": "11MA1", "lehrer": "Doe", "raum": "R101" },
    ...
  ]
}
```

## Verwendung der Beispieldaten

1. Starten Sie die Anwendung mit `python start.py`
2. Navigieren Sie zum Daten-Upload-Bereich
3. Laden Sie die gewünschten Beispieldateien hoch
4. Oder verwenden Sie die vorinstallierten Beispieldaten über die entsprechende Option

## Eigene Daten

Wenn Sie Ihre eigenen Daten verwenden möchten, stellen Sie bitte sicher, dass diese dem oben beschriebenen Format entsprechen.
