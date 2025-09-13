import base64
import csv
import hashlib
import io
import json
import logging
import os
import pickle
import re
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import uuid
import warnings
import webbrowser
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from io import BytesIO, StringIO

import extra_streamlit_components as stx
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import py3Dmol
import pymongo
import requests
import seaborn as sns
import streamlit as st
import streamlit_analytics
import streamlit_toggle as tog
import xlsxwriter
from htbuilder import big, div, h2, styles
from htbuilder.units import rem
from PIL import Image
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from stqdm import stqdm
from streamlit_ace import st_ace
from streamlit_authenticator import Authenticate

# Eigene Module importieren
from visualization_utils import (
    calculate_course_student_avg,
    plot_course_student_counts,
    search_students_by_name,
    search_students_by_courses,
    calculate_student_overlap,
    plot_student_overlap_heatmap,
    generate_student_overlap_svg,
    create_person_network_graph,
    create_student_course_sankey,
    create_3d_student_network,
    perform_student_clustering,
    generate_pdf_report,
    generate_qr_code,
    anonymize_student_data,
    export_plotly_figure,
    AuditLogger
)
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.app_logo import add_logo
from streamlit_extras.card import card
from streamlit_extras.chart_container import chart_container
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
# Fortgeschrittene UI-Komponenten
from streamlit_extras.switch_page_button import switch_page
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# Visualisierungsfunktionen importieren
from visualization_utils import (analyze_teacher_workload,
                               calculate_course_overlap,
                               create_3d_course_visualization,
                               create_network_graph,
                               create_person_network_graph,
                               create_timetable_heatmap,
                               generate_participant_course_matrix,
                               generate_student_overlap_svg,
                               generate_student_timetable, 
                               get_room_usage,
                               plot_course_overlap_heatmap,
                               plot_course_participants_bar,
                               plot_student_overlap_heatmap,
                               search_students_by_courses,
                               search_students_by_name,
                               calculate_student_overlap,
                               export_plotly_figure,
                               visualize_student_timetable)

# Leistungsoptimierungen
warnings.filterwarnings('ignore')
try:
    # Option für ältere Streamlit-Versionen, bei neueren ignorieren
    st.set_option('deprecation.showPyplotGlobalUse', False)
except Exception:
    pass

# Eigenes Log-Level für Performance-Messung
PERF = 15  # Zwischen DEBUG und INFO
logging.addLevelName(PERF, "PERF")

# Methode zum Loggen von Performance-Metriken
def perf(self, message, *args, **kwargs):
    if self.isEnabledFor(PERF):
        self._log(PERF, message, args, **kwargs)

# Hinzufügen der perf-Methode zu Logger-Klasse
logging.Logger.perf = perf

# Stellen Sie sicher, dass das logs-Verzeichnis existiert
os.makedirs("logs", exist_ok=True)

# RotatingFileHandler für bessere Logfile-Verwaltung
try:
    from logging.handlers import RotatingFileHandler
    log_handler = RotatingFileHandler(
        "logs/app.log", 
        mode='a', 
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,  # 5 Backup-Dateien
        encoding='utf-8'
    )
except ImportError:
    # Fallback, wenn RotatingFileHandler nicht verfügbar ist
    log_handler = logging.FileHandler(
        "logs/app.log", 
        mode='a',
        encoding='utf-8'
    )

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        log_handler,
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Globale Konfigurationen
VERSION = "2.0.0"
CACHE_TTL = 3600  # Cache-Gültigkeit in Sekunden
THEME_COLOR = "#2563EB"
DEFAULT_DB_PATH = "data/app.db"
MONGODB_URI = "mongodb://localhost:27017/"

# PWA-Manifest für Offline-Nutzung
PWA_MANIFEST = {
    "name": "Kursplan-Analyse",
    "short_name": "KursAnalyse",
    "description": "Umfassende Analyse von Stunden- und Kursplänen",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#ffffff",
    "theme_color": THEME_COLOR,
    "orientation": "portrait-primary",
    "icons": [
        {
            "src": "icon-192x192.png",
            "sizes": "192x192",
            "type": "image/png",
            "purpose": "any maskable"
        },
        {
            "src": "icon-512x512.png",
            "sizes": "512x512",
            "type": "image/png",
            "purpose": "any maskable"
        }
    ]
}

# Globaler Cache-Speicher
cache = {}
cache_version = int(time.time())

# Farbpalette für Visualisierungen
COLOR_PALETTE = {
    "primary": "#2563EB",     # Blau
    "secondary": "#7C3AED",   # Lila
    "tertiary": "#10B981",    # Grün
    "warning": "#F59E0B",     # Orange
    "danger": "#EF4444",      # Rot
    "info": "#3B82F6",        # Hellblau
    "success": "#10B981",     # Hellgrün
    "light": "#F3F4F6",       # Hellgrau
    "dark": "#1F2937",        # Dunkelgrau
    "white": "#FFFFFF",       # Weiß
    "black": "#000000",       # Schwarz
    "background": "#F9FAFB",  # Hintergrund
    "text": "#111827",        # Text
    "muted": "#6B7280"        # Gedämpfter Text
}

# Performance-Tracking
performance_metrics = {
    "page_load_time": [],
    "data_load_time": [],
    "rendering_time": [],
    "total_time": []
}

# Seitenkonfiguration
st.set_page_config(
    page_title="Kursplan-Analyse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/yourusername/kursplan-analyse/issues",
        "Report a bug": "https://github.com/yourusername/kursplan-analyse/issues/new",
        "About": f"# Kursplan-Analyse v{VERSION}\nEine umfassende Analyse-App für Stunden- und Kurspläne."
    }
)

# Hilfsklassen für die Anwendung
class AppState:
    """Globaler Anwendungszustand für Sitzungsverwaltung"""
    
    def __init__(self):
        self.is_authenticated = False
        self.user = None
        self.settings = {}
        self.start_time = time.time()
        self.db_connections = {}
        self.last_action = None
        self.action_history = []
        self.data_cache = {}
        self.session_id = str(uuid.uuid4())
    
    def track_action(self, action_name, metadata=None):
        """Verfolgt Benutzeraktionen für Analysen"""
        timestamp = time.time()
        action = {
            "action": action_name,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        self.last_action = action
        self.action_history.append(action)
        logger.info(f"Action tracked: {action_name}")
    
    def get_session_duration(self):
        """Gibt die Dauer der aktuellen Sitzung zurück"""
        return time.time() - self.start_time
    
    def clear_cache(self):
        """Löscht den Cache"""
        self.data_cache = {}
        global cache, cache_version
        cache = {}
        cache_version = int(time.time())
        st.cache_data.clear()
        logger.info("Cache cleared")
    
    def save_settings(self):
        """Speichert Benutzereinstellungen"""
        try:
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=4)
            logger.info("Settings saved")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def load_settings(self):
        """Lädt Benutzereinstellungen"""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                logger.info("Settings loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return False

class Database:
    """Klasse für Datenbankoperationen"""
    
    def __init__(self, db_type="sqlite", connection_string=None):
        self.db_type = db_type
        self.connection_string = connection_string or DEFAULT_DB_PATH
        self.connection = None
        self.is_connected = False
    
    def connect(self):
        """Stellt eine Verbindung zur Datenbank her"""
        try:
            if self.db_type == "sqlite":
                self.connection = sqlite3.connect(self.connection_string)
                self.is_connected = True
                logger.info(f"Connected to SQLite: {self.connection_string}")
                return True
            elif self.db_type == "mongodb":
                self.connection = pymongo.MongoClient(self.connection_string)
                self.is_connected = True
                logger.info(f"Connected to MongoDB: {self.connection_string}")
                return True
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Trennt die Verbindung zur Datenbank"""
        try:
            if self.is_connected:
                if self.db_type == "sqlite":
                    self.connection.close()
                elif self.db_type == "mongodb":
                    self.connection.close()
                self.is_connected = False
                logger.info("Database disconnected")
                return True
            return False
        except Exception as e:
            logger.error(f"Database disconnect error: {e}")
            return False
    
    def save_dataframe(self, df, table_name, if_exists="replace"):
        """Speichert ein DataFrame in der Datenbank"""
        try:
            if not self.is_connected:
                self.connect()
            
            if self.db_type == "sqlite":
                df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
                logger.info(f"DataFrame saved to SQLite table: {table_name}")
                return True
            elif self.db_type == "mongodb":
                db = self.connection["kursplan_analyse"]
                collection = db[table_name]
                
                # Konvertiere DataFrame zu Dict-Liste
                records = df.to_dict("records")
                
                if if_exists == "replace":
                    collection.delete_many({})
                
                collection.insert_many(records)
                logger.info(f"DataFrame saved to MongoDB collection: {table_name}")
                return True
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
        except Exception as e:
            logger.error(f"Error saving DataFrame to database: {e}")
            return False
    
    def load_dataframe(self, table_name, query=None):
        """Lädt ein DataFrame aus der Datenbank"""
        try:
            if not self.is_connected:
                self.connect()
            
            if self.db_type == "sqlite":
                if query:
                    return pd.read_sql_query(query, self.connection)
                else:
                    return pd.read_sql_query(f"SELECT * FROM {table_name}", self.connection)
            elif self.db_type == "mongodb":
                db = self.connection["kursplan_analyse"]
                collection = db[table_name]
                
                if query:
                    cursor = collection.find(query)
                else:
                    cursor = collection.find()
                
                df = pd.DataFrame(list(cursor))
                
                # MongoDB-ID entfernen, falls vorhanden
                if "_id" in df.columns:
                    df = df.drop("_id", axis=1)
                
                return df
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading DataFrame from database: {e}")
            return pd.DataFrame()

class DataLoader:
    """Klasse für das Laden und Verarbeiten von Daten"""
    
    def __init__(self, app_state):
        self.app_state = app_state
    
    @st.cache_data(ttl=CACHE_TTL)
    def load_json_file(self, file_path):
        """Lädt eine JSON-Datei mit Caching"""
        try:
            cache_key = f"json_{file_path}_{cache_version}"
            
            if cache_key in cache:
                logger.info(f"Cache hit for {file_path}")
                return cache[cache_key]
            
            start_time = time.time()
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {file_path} in {load_time:.2f} seconds")
            
            cache[cache_key] = data
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            st.error(f"Datei nicht gefunden: {file_path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in file: {file_path}")
            st.error(f"Ungültiges JSON-Format in der Datei: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            st.error(f"Fehler beim Laden der Datei {file_path}: {e}")
            return None
    
    def load_csv_file(self, file_path, **kwargs):
        """Lädt eine CSV-Datei mit Caching"""
        try:
            cache_key = f"csv_{file_path}_{cache_version}"
            
            if cache_key in cache:
                logger.info(f"Cache hit for {file_path}")
                return cache[cache_key]
            
            start_time = time.time()
            df = pd.read_csv(file_path, **kwargs)
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {file_path} in {load_time:.2f} seconds")
            
            cache[cache_key] = df
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            st.error(f"Fehler beim Laden der CSV-Datei {file_path}: {e}")
            return None
    
    def save_uploaded_file(self, uploaded_file, target_dir="uploaded_files"):
        """Speichert eine hochgeladene Datei"""
        try:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            file_path = os.path.join(target_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"File saved: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            st.error(f"Fehler beim Speichern der hochgeladenen Datei: {e}")
            return None
    
    def process_uploaded_file(self, uploaded_file):
        """Verarbeitet eine hochgeladene Datei basierend auf dem Dateityp"""
        try:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            if file_type == "json":
                data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                return data
            elif file_type == "csv":
                df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
                return df
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file)
                return df
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                st.warning(f"Nicht unterstützter Dateityp: {file_type}")
                return None
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            st.error(f"Fehler bei der Verarbeitung der hochgeladenen Datei: {e}")
            return None

class Analytics:
    """Klasse für Analysen und Datenverarbeitung"""
    
    def __init__(self, app_state):
        self.app_state = app_state
    
    @st.cache_data(ttl=CACHE_TTL)
    def convert_timetable_to_df(self, timetable_data):
        """Konvertiert Stundenplan-Daten in DataFrame mit Parallelisierung für große Datensätze"""
        try:
            if not timetable_data:
                return pd.DataFrame()
            
            start_time = time.time()
            
            # Für große Datensätze Parallelisierung verwenden
            if len(timetable_data) > 1000:
                with ThreadPoolExecutor() as executor:
                    # Daten in Chunks aufteilen
                    chunk_size = max(1, len(timetable_data) // os.cpu_count())
                    chunks = [timetable_data[i:i+chunk_size] for i in range(0, len(timetable_data), chunk_size)]
                    
                    # Verarbeitung parallel ausführen
                    results = list(executor.map(self._process_timetable_chunk, chunks))
                
                # Ergebnisse zusammenführen
                timetable_rows = []
                for result in results:
                    timetable_rows.extend(result)
            else:
                # Für kleinere Datensätze sequentiell verarbeiten
                timetable_rows = self._process_timetable_chunk(timetable_data)
            
            df = pd.DataFrame(timetable_rows)
            
            # Optimierung für bessere Filterung und Sortierung
            if not df.empty and 'Tag' in df.columns:
                # Wochentage in richtige Reihenfolge bringen
                weekday_order = {
                    'Montag': 0, 
                    'Dienstag': 1, 
                    'Mittwoch': 2, 
                    'Donnerstag': 3, 
                    'Freitag': 4
                }
                
                df['Tag_Order'] = df['Tag'].map(weekday_order)
                df = df.sort_values(['Tag_Order', 'Stunde']).drop('Tag_Order', axis=1)
            
            process_time = time.time() - start_time
            logger.info(f"Converted timetable data to DataFrame in {process_time:.2f} seconds")
            
            return df
        except Exception as e:
            logger.error(f"Error converting timetable data: {e}")
            return pd.DataFrame()
    
    def _process_timetable_chunk(self, chunk):
        """Verarbeitet einen Chunk von Stundenplan-Daten"""
        timetable_rows = []
        
        for entry in chunk:
            tag = entry.get('tag', '')
            stunde = entry.get('stunde', 0)
            
            faecher = entry.get('faecher', [])
            
            # Für 8. und 9. Stunde: Wenn keine Fächer, keinen Eintrag hinzufügen
            if stunde in [8, 9] and not faecher:
                continue
                
            if not faecher:  # Leere Stunde (aber nicht 8./9. Stunde)
                timetable_rows.append({
                    'Tag': tag,
                    'Stunde': stunde,
                    'Fach': '',
                    'Lehrer': '',
                    'Raum': ''
                })
            else:
                for fach_entry in faecher:
                    timetable_rows.append({
                        'Tag': tag,
                        'Stunde': stunde,
                        'Fach': fach_entry.get('fach', ''),
                        'Lehrer': fach_entry.get('lehrer', ''),
                        'Raum': fach_entry.get('raum', '')
                    })
        
        return timetable_rows
    
    @st.cache_data(ttl=CACHE_TTL)
    def convert_courses_to_df(self, courses_data):
        """Konvertiert Kursdaten in DataFrame mit Optimierungen"""
        try:
            if not courses_data:
                return pd.DataFrame()
            
            start_time = time.time()
            
            course_rows = []
            for course in courses_data:
                kurs_id = course.get('kurs_id', '')
                kurs = course.get('kurs', '')
                kurs_type = course.get('kurs_type', '')
                participants_count = course.get('participants_count', 0)
                participants = course.get('participants', [])
                
                course_rows.append({
                    'Kurs_ID': kurs_id,
                    'Kurs': kurs,
                    'Kurstyp': kurs_type,
                    'Teilnehmerzahl': participants_count,
                    'Teilnehmer': participants
                })
            
            df = pd.DataFrame(course_rows)
            
            process_time = time.time() - start_time
            logger.info(f"Converted course data to DataFrame in {process_time:.2f} seconds")
            
            return df
        except Exception as e:
            logger.error(f"Error converting course data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=CACHE_TTL)
    def convert_students_to_df(self, students_data):
        """Konvertiert Schülerdaten in DataFrame mit erweiterten Features"""
        try:
            if not students_data:
                return pd.DataFrame()
            
            start_time = time.time()
            
            student_rows = []
            for student in students_data:
                student_id = student.get('student_id', '')
                name = student.get('name', '')
                courses_str = student.get('courses', '')
                
                # Kursliste extrahieren
                courses_list = self._parse_student_courses(courses_str)
                
                student_rows.append({
                    'Schüler_ID': student_id,
                    'Name': name,
                    'Kurse_String': courses_str,
                    'Kurse_Liste': courses_list,
                    'Anzahl_Kurse': len(courses_list)
                })
            
            df = pd.DataFrame(student_rows)
            
            process_time = time.time() - start_time
            logger.info(f"Converted student data to DataFrame in {process_time:.2f} seconds")
            
            return df
        except Exception as e:
            logger.error(f"Error converting student data: {e}")
            return pd.DataFrame()
    
    def _parse_student_courses(self, courses_str):
        """Konvertiert einen Kurs-String in eine Liste von Kursen"""
        if not courses_str:
            return []
        return [course.strip() for course in courses_str.split(',')]
    
    @st.cache_data(ttl=CACHE_TTL)
    def generate_participant_course_matrix(self, students_df):
        """Erstellt eine Matrix, die zeigt, welche Schüler in welchen Kursen sind"""
        try:
            if students_df.empty:
                return pd.DataFrame()
            
            start_time = time.time()
            
            # Alle Kurse aus den Schülerdaten extrahieren
            all_courses = set()
            for courses_list in students_df['Kurse_Liste']:
                all_courses.update(courses_list)
            
            # Matrix erstellen: Zeilen = Schüler, Spalten = Kurse
            matrix_data = {course: [] for course in sorted(all_courses)}
            
            for _, student in students_df.iterrows():
                for course in all_courses:
                    matrix_data[course].append(1 if course in student['Kurse_Liste'] else 0)
            
            # DataFrame erstellen
            df = pd.DataFrame(matrix_data, index=students_df['Name'])
            
            process_time = time.time() - start_time
            logger.info(f"Generated participant-course matrix in {process_time:.2f} seconds")
            
            return df
        except Exception as e:
            logger.error(f"Error generating participant-course matrix: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=CACHE_TTL)
    def calculate_course_overlap(self, participant_course_matrix):
        """Berechnet die Überschneidungen zwischen Kursen mit Optimierungen"""
        try:
            if participant_course_matrix.empty:
                return pd.DataFrame()
            
            start_time = time.time()
            
            courses = participant_course_matrix.columns
            overlap_matrix = pd.DataFrame(index=courses, columns=courses)
            
            # Für jeden Kurs die Studenten finden, die ihn belegen
            course_students = {}
            for course in courses:
                course_students[course] = set(participant_course_matrix.index[participant_course_matrix[course] == 1])
            
            # Überschneidungen berechnen
            for i, course1 in enumerate(courses):
                students_in_course1 = course_students[course1]
                
                # Diagonale direkt setzen
                overlap_matrix.loc[course1, course1] = len(students_in_course1)
                
                # Nur die obere Dreiecksmatrix berechnen, dann spiegeln
                for j in range(i+1, len(courses)):
                    course2 = courses[j]
                    students_in_course2 = course_students[course2]
                    
                    overlap = len(students_in_course1.intersection(students_in_course2))
                    overlap_matrix.loc[course1, course2] = overlap
                    overlap_matrix.loc[course2, course1] = overlap  # Symmetrische Matrix
            
            process_time = time.time() - start_time
            logger.info(f"Calculated course overlap matrix in {process_time:.2f} seconds")
            
            return overlap_matrix
        except Exception as e:
            logger.error(f"Error calculating course overlap: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=CACHE_TTL)
    def analyze_teacher_workload(self, timetable_df, courses_df=None, course_details_df=None):
        """Analysiert die Arbeitsbelastung der Lehrer mit erweiterten Metriken"""
        try:
            if timetable_df.empty:
                return pd.DataFrame()
            
            start_time = time.time()
            
            # Grundlegende Arbeitsbelastung aus dem Stundenplan
            teacher_workload = timetable_df.groupby('Lehrer').size().reset_index(name='Anzahl_Stunden')
            teacher_workload = teacher_workload[teacher_workload['Lehrer'] != '']  # Leere Werte ausschließen
            
            # Wenn Kursdetails verfügbar sind, diese hinzufügen
            if course_details_df is not None and not course_details_df.empty:
                # Anzahl der Kurse pro Lehrer
                teacher_courses = course_details_df.groupby('Lehrer_Kuerzel').size().reset_index(name='Anzahl_Kurse')
                
                # Lehrerdetails hinzufügen
                teacher_details = course_details_df[['Lehrer_Kuerzel', 'Lehrer_Name']].drop_duplicates()
                
                # Mit Arbeitsbelastungsdaten zusammenführen
                teacher_workload = pd.merge(
                    teacher_workload,
                    teacher_details,
                    left_on='Lehrer',
                    right_on='Lehrer_Kuerzel',
                    how='left'
                )
                
                # Anzahl der Kurse hinzufügen
                teacher_workload = pd.merge(
                    teacher_workload,
                    teacher_courses,
                    left_on='Lehrer',
                    right_on='Lehrer_Kuerzel',
                    how='left'
                )
                
                # Fehlende Werte auffüllen
                teacher_workload['Anzahl_Kurse'] = teacher_workload['Anzahl_Kurse'].fillna(0).astype(int)
                
                # Spalten bereinigen
                if 'Lehrer_Kuerzel_x' in teacher_workload.columns:
                    teacher_workload = teacher_workload.drop(['Lehrer_Kuerzel_x', 'Lehrer_Kuerzel_y'], axis=1)
                elif 'Lehrer_Kuerzel' in teacher_workload.columns:
                    teacher_workload = teacher_workload.drop('Lehrer_Kuerzel', axis=1)
            
            # Wenn Kursdetails und Teilnehmerdaten verfügbar sind, weitere Analysen hinzufügen
            if courses_df is not None and not courses_df.empty and course_details_df is not None and not course_details_df.empty:
                # Kurse mit Teilnehmerzahlen
                course_participants = courses_df[['Kurs', 'Teilnehmerzahl']]
                
                # Mit Kursdetails zusammenführen
                course_with_teachers = pd.merge(
                    course_participants,
                    course_details_df[['Kurs', 'Lehrer_Kuerzel']],
                    on='Kurs',
                    how='inner'
                )
                
                # Gesamtzahl der Schüler pro Lehrer
                teacher_students = course_with_teachers.groupby('Lehrer_Kuerzel')['Teilnehmerzahl'].sum().reset_index(
                    name='Gesamtzahl_Schüler')
                
                # Mit Arbeitsbelastungsdaten zusammenführen
                teacher_workload = pd.merge(
                    teacher_workload,
                    teacher_students,
                    left_on='Lehrer',
                    right_on='Lehrer_Kuerzel',
                    how='left'
                )
                
                # Fehlende Werte auffüllen
                teacher_workload['Gesamtzahl_Schüler'] = teacher_workload['Gesamtzahl_Schüler'].fillna(0).astype(int)
                
                # Spalten bereinigen
                if 'Lehrer_Kuerzel' in teacher_workload.columns:
                    teacher_workload = teacher_workload.drop('Lehrer_Kuerzel', axis=1)
                
                # Durchschnittliche Schülerzahl pro Kurs berechnen
                teacher_workload['Durchschnitt_Schüler_pro_Kurs'] = (
                    teacher_workload['Gesamtzahl_Schüler'] / 
                    teacher_workload['Anzahl_Kurse'].replace(0, np.nan)
                ).fillna(0)
                
                # Belastungsindex berechnen (gewichtete Summe aus Stunden, Kursen und Schülern)
                teacher_workload['Belastungsindex'] = (
                    teacher_workload['Anzahl_Stunden'] * 1.0 + 
                    teacher_workload['Anzahl_Kurse'] * 1.5 + 
                    teacher_workload['Gesamtzahl_Schüler'] * 0.1
                )
            
            # Nach Arbeitsbelastung sortieren
            teacher_workload = teacher_workload.sort_values('Anzahl_Stunden', ascending=False)
            
            process_time = time.time() - start_time
            logger.info(f"Analyzed teacher workload in {process_time:.2f} seconds")
            
            return teacher_workload
        except Exception as e:
            logger.error(f"Error analyzing teacher workload: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=CACHE_TTL)
    def get_room_usage(self, timetable_df):
        """Analysiert die Raumnutzung mit detaillierten Statistiken"""
        try:
            if timetable_df.empty:
                return pd.DataFrame()
            
            start_time = time.time()
            
            # Grundlegende Raumnutzung
            room_usage = timetable_df.groupby('Raum').size().reset_index(name='Nutzungshäufigkeit')
            room_usage = room_usage[room_usage['Raum'] != '']  # Leere Werte ausschließen
            
            # Raumnutzung nach Tagen
            room_usage_by_day = timetable_df.groupby(['Raum', 'Tag']).size().reset_index(name='Stunden_pro_Tag')
            
            # Durchschnittliche, minimale und maximale Nutzung pro Tag
            room_stats = room_usage_by_day.groupby('Raum').agg(
                Durchschnitt_pro_Tag=('Stunden_pro_Tag', 'mean'),
                Minimum_pro_Tag=('Stunden_pro_Tag', 'min'),
                Maximum_pro_Tag=('Stunden_pro_Tag', 'max')
            ).reset_index()
            
            # Mit Gesamtnutzung zusammenführen
            room_usage = pd.merge(room_usage, room_stats, on='Raum', how='left')
            
            # Auslastung berechnen (angenommen, max. 9 Stunden pro Tag, 5 Tage)
            max_possible_hours = 9 * 5
            room_usage['Auslastung_Prozent'] = (room_usage['Nutzungshäufigkeit'] / max_possible_hours) * 100
            
            # Nach Nutzungshäufigkeit sortieren
            room_usage = room_usage.sort_values('Nutzungshäufigkeit', ascending=False)
            
            process_time = time.time() - start_time
            logger.info(f"Analyzed room usage in {process_time:.2f} seconds")
            
            return room_usage
        except Exception as e:
            logger.error(f"Error analyzing room usage: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=CACHE_TTL)
    def generate_student_timetable(self, student_name, students_df, timetable_df, course_participants_df=None):
        """Generiert einen individuellen Stundenplan für einen Schüler"""
        try:
            if students_df.empty or timetable_df.empty:
                return pd.DataFrame()
            
            start_time = time.time()
            
            # Kurse des Schülers finden
            student_data = students_df[students_df['Name'] == student_name]
            if student_data.empty:
                return pd.DataFrame()
            
            student_courses = student_data.iloc[0]['Kurse_Liste']
            
            # Stundenplan filtern auf die Kurse des Schülers
            student_timetable = timetable_df[timetable_df['Fach'].isin(student_courses)]
            
            # Leeres DataFrame für alle Tage und Stunden erstellen
            weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag']
            hours = range(0, 10)  # 0 bis 9. Stunde
            
            full_timetable = pd.DataFrame(
                [(day, hour) for day in weekdays for hour in hours],
                columns=['Tag', 'Stunde']
            )
            
            # Mit dem Stundenplan des Schülers zusammenführen
            merged_timetable = pd.merge(
                full_timetable,
                student_timetable,
                on=['Tag', 'Stunde'],
                how='left'
            )
            
            # Nach Tag und Stunde sortieren
            weekday_order = {
                'Montag': 0, 
                'Dienstag': 1, 
                'Mittwoch': 2, 
                'Donnerstag': 3, 
                'Freitag': 4
            }
            
            merged_timetable['Tag_Order'] = merged_timetable['Tag'].map(weekday_order)
            merged_timetable = merged_timetable.sort_values(['Tag_Order', 'Stunde']).drop('Tag_Order', axis=1)
            
            process_time = time.time() - start_time
            logger.info(f"Generated student timetable in {process_time:.2f} seconds")
            
            return merged_timetable
        except Exception as e:
            logger.error(f"Error generating student timetable: {e}")
            return pd.DataFrame()

class Visualization:
    """Klasse für Datenvisualisierungen"""
    
    def __init__(self, app_state):
        self.app_state = app_state
        self.color_palette = COLOR_PALETTE
    
    def plot_course_overlap_heatmap(self, overlap_matrix):
        """Erstellt eine interaktive Heatmap der Kursüberschneidungen"""
        try:
            if overlap_matrix.empty:
                return None
            
            start_time = time.time()
            
            fig = px.imshow(
                overlap_matrix,
                labels=dict(x="Kurs", y="Kurs", color="Anzahl überschneidender Schüler"),
                x=overlap_matrix.columns,
                y=overlap_matrix.index,
                color_continuous_scale='YlGnBu',
                title='Überschneidungen zwischen Kursen (Anzahl der gemeinsamen Schüler)'
            )
            
            fig.update_layout(
                height=800,
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis={'side': 'bottom', 'tickangle': -45},
                yaxis={'autorange': 'reversed'},
                coloraxis_colorbar=dict(
                    title="Anzahl Schüler",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=300,
                ),
                font=dict(family="Open Sans, sans-serif"),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Open Sans, sans-serif"
                )
            )
            
            render_time = time.time() - start_time
            logger.info(f"Rendered course overlap heatmap in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting course overlap heatmap: {e}")
            return None
    
    def plot_course_participants_bar(self, courses_df, num_courses=None, highlight_course=None):
        """Erstellt ein interaktives Balkendiagramm für die Anzahl der Teilnehmer pro Kurs"""
        try:
            if courses_df.empty:
                return None
            
            start_time = time.time()
            
            if num_courses:
                courses_to_plot = courses_df.sort_values('Teilnehmerzahl', ascending=False).head(num_courses)
            else:
                courses_to_plot = courses_df
            
            # Farben anpassen, wenn ein Kurs hervorgehoben werden soll
            if highlight_course:
                colors = [self.color_palette['secondary'] if x != highlight_course else self.color_palette['danger'] 
                         for x in courses_to_plot['Kurs']]
                color_discrete_map = None
            else:
                colors = None
                color_discrete_map = {
                    'Leistungskurs': self.color_palette['primary'],
                    'Grundkurs': self.color_palette['secondary']
                }
            
            fig = px.bar(
                courses_to_plot, 
                x='Kurs', 
                y='Teilnehmerzahl',
                color='Kurstyp' if not highlight_course else None,
                color_discrete_map=color_discrete_map,
                title='Teilnehmeranzahl pro Kurs',
                labels={'Teilnehmerzahl': 'Anzahl der Teilnehmer', 'Kurs': 'Kursbezeichnung'},
                text='Teilnehmerzahl',
                hover_data=['Kurstyp']
            )
            
            if highlight_course:
                fig.update_traces(marker_color=colors)
            
            fig.update_layout(
                xaxis_tickangle=-45,
                xaxis={'categoryorder':'total descending'},
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Open Sans, sans-serif"
                ),
                uniformtext_minsize=10, 
                uniformtext_mode='hide',
                font=dict(family="Open Sans, sans-serif")
            )
            
            render_time = time.time() - start_time
            logger.info(f"Rendered course participants bar chart in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting course participants bar chart: {e}")
            return None
    
    def create_timetable_heatmap(self, timetable_df):
        """Erstellt eine interaktive Heatmap des Stundenplans"""
        try:
            if timetable_df.empty:
                return None
            
            start_time = time.time()
            
            # Aggregiere die Anzahl der Fächer pro Tag und Stunde
            heatmap_data = timetable_df.groupby(['Tag', 'Stunde']).size().reset_index(name='Anzahl_Faecher')
            
            # Konvertiere zu einer Pivot-Tabelle für die Heatmap
            pivot_data = heatmap_data.pivot(index='Stunde', columns='Tag', values='Anzahl_Faecher').fillna(0)
            
            # Wochentage in richtige Reihenfolge bringen
            wochentage = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag']
            pivot_data = pivot_data.reindex(columns=wochentage)
            
            # Heatmap erstellen
            fig = px.imshow(
                pivot_data,
                labels=dict(x="Tag", y="Stunde", color="Anzahl Fächer"),
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale='YlGnBu',
                title='Stundenplanauslastung',
                text_auto=True
            )
            
            fig.update_layout(
                xaxis={'side': 'top'},
                yaxis={'dtick': 1},
                coloraxis_colorbar=dict(
                    title="Anzahl Fächer",
                    thicknessmode="pixels", thickness=20,
                    lenmode="pixels", len=300,
                ),
                font=dict(family="Open Sans, sans-serif"),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Open Sans, sans-serif"
                )
            )
            
            render_time = time.time() - start_time
            logger.info(f"Rendered timetable heatmap in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating timetable heatmap: {e}")
            return None
    
    def create_network_graph(self, participant_course_matrix, threshold=3, focus_entity=None):
        """Erstellt ein interaktives Netzwerkdiagramm für Schüler-Kurs-Beziehungen"""
        try:
            if participant_course_matrix.empty:
                return None
            
            start_time = time.time()
            
            # Netzwerk erstellen
            G = nx.Graph()
            
            # Kurse als Knoten hinzufügen (mit Blau)
            for course in participant_course_matrix.columns:
                G.add_node(course, bipartite=0, type='course', size=10)
            
            # Schüler als Knoten hinzufügen (mit Rot)
            for student in participant_course_matrix.index:
                G.add_node(student, bipartite=1, type='student', size=5)
            
            # Kanten zwischen Schülern und Kursen hinzufügen
            for student in participant_course_matrix.index:
                for course in participant_course_matrix.columns:
                    if participant_course_matrix.loc[student, course] == 1:
                        G.add_edge(student, course, weight=1)
            
            # Wenn ein Fokus-Element angegeben ist, nur verwandte Knoten anzeigen
            if focus_entity:
                if focus_entity in G:
                    # Nachbarn des Fokus-Elements
                    neighbors = list(G.neighbors(focus_entity))
                    # Für jeden Nachbarn auch dessen Nachbarn hinzufügen (für Kurs-Schüler-Kurs-Verbindungen)
                    second_neighbors = []
                    for neighbor in neighbors:
                        second_neighbors.extend(list(G.neighbors(neighbor)))
                    # Alle relevanten Knoten
                    nodes_to_keep = [focus_entity] + neighbors + second_neighbors
                    G = G.subgraph(nodes_to_keep)
            
            # Layout berechnen
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            # Knoten und Kanten für Plotly vorbereiten
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                if G.nodes[node]['type'] == 'course':
                    node_color.append(self.color_palette['primary'])  # Blau für Kurse
                    node_size.append(15)
                else:
                    node_color.append(self.color_palette['danger'])  # Rot für Schüler
                    node_size.append(10)
            
            # Kanten vorbereiten
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Kanten zeichnen
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#94A3B8'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Knoten zeichnen
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=node_size,
                    line=dict(width=1, color='#CBD5E1')
                ),
                text=node_text,
                hoverinfo='text'
            )
            
            # Layout erstellen
            layout = go.Layout(
                title='Schüler-Kurs-Netzwerk',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Open Sans, sans-serif"),
                annotations=[
                    dict(
                        text="Blau: Kurse | Rot: Schüler",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.01, y=-0.01,
                        font=dict(family="Open Sans, sans-serif")
                    )
                ]
            )
            
            # Figure erstellen
            fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
            
            render_time = time.time() - start_time
            logger.info(f"Rendered network graph in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating network graph: {e}")
            return None
    
    def create_3d_course_visualization(self, overlap_matrix):
        """Erstellt eine 3D-Visualisierung der Kursüberschneidungen"""
        try:
            if overlap_matrix.empty:
                return None
            
            start_time = time.time()
            
            # Diagonale (Selbstüberschneidung) auf 0 setzen
            np.fill_diagonal(overlap_matrix.values, 0)
            
            # 3D-Scatter-Plot vorbereiten
            fig = go.Figure()
            
            # Multi-dimensional scaling (MDS) für 3D-Koordinaten
            from sklearn.manifold import MDS
            mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
            
            # Distanzmatrix erstellen (inverse der Überlappung)
            max_overlap = overlap_matrix.max().max()
            if max_overlap > 0:
                distance_matrix = 1 - (overlap_matrix / max_overlap)
                positions = mds.fit_transform(distance_matrix)
            else:
                # Wenn keine Überlappungen, zufällige Positionen verwenden
                positions = np.random.rand(len(overlap_matrix), 3)
            
            # Cluster basierend auf Überlappungen
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=min(5, len(overlap_matrix)),
                affinity='precomputed',
                linkage='average'
            ).fit(distance_matrix)
            
            # Farben und Größen basierend auf Clustern und Überlappungssummen
            colors = [
                self.color_palette['primary'],
                self.color_palette['secondary'],
                self.color_palette['danger'],
                self.color_palette['success'],
                self.color_palette['warning']
            ]
            sizes = overlap_matrix.sum(axis=1).values  # Gesamte Überlappung pro Kurs
            max_size = max(sizes) if len(sizes) > 0 else 1
            normalized_sizes = [20 + (s / max_size) * 30 for s in sizes]  # Größen zwischen 20 und 50
            
            # 3D-Scatter-Plot erstellen
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers+text',
                marker=dict(
                    size=normalized_sizes,
                    color=[colors[c % len(colors)] for c in clustering.labels_],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                text=overlap_matrix.index,
                hoverinfo='text',
                hovertext=[f"Kurs: {course}<br>Gesamtüberlappung: {int(sizes[i])}" 
                          for i, course in enumerate(overlap_matrix.index)]
            ))
            
            # Layout anpassen
            fig.update_layout(
                title='3D-Visualisierung der Kursüberschneidungen',
                scene=dict(
                    xaxis=dict(showticklabels=False, title=''),
                    yaxis=dict(showticklabels=False, title=''),
                    zaxis=dict(showticklabels=False, title=''),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700,
                font=dict(family="Open Sans, sans-serif")
            )
            
            render_time = time.time() - start_time
            logger.info(f"Rendered 3D course visualization in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating 3D course visualization: {e}")
            return None
    
    def plot_teacher_workload(self, teacher_workload, metric='Anzahl_Stunden', top_n=10):
        """Erstellt ein Balkendiagramm für die Lehrerbelastung"""
        try:
            if teacher_workload.empty:
                return None
            
            start_time = time.time()
            
            # Metriklabels für die Anzeige
            metric_labels = {
                'Anzahl_Stunden': 'Anzahl der Unterrichtsstunden',
                'Anzahl_Kurse': 'Anzahl der unterrichteten Kurse',
                'Gesamtzahl_Schüler': 'Gesamtzahl der Schüler',
                'Belastungsindex': 'Belastungsindex'
            }
            
            # Sortieren nach gewählter Metrik
            sorted_data = teacher_workload.sort_values(metric, ascending=False).head(top_n)
            
            # Spalte für die Anzeige im Diagramm auswählen
            display_name = 'Lehrer'
            if 'Lehrer_Name' in sorted_data.columns:
                sorted_data['Anzeigename'] = sorted_data.apply(
                    lambda row: f"{row['Lehrer_Name']} ({row['Lehrer']})" if pd.notna(row['Lehrer_Name']) else row['Lehrer'], 
                    axis=1
                )
                display_name = 'Anzeigename'
            
            # Diagramm erstellen
            fig = px.bar(
                sorted_data,
                x=display_name,
                y=metric,
                title=f'Top {top_n} Lehrer nach {metric_labels.get(metric, metric)}',
                labels={metric: metric_labels.get(metric, metric), display_name: 'Lehrer'},
                color=metric,
                color_continuous_scale='Viridis'
            )
            
            # Layout anpassen
            fig.update_layout(
                xaxis_tickangle=-45,
                font=dict(family="Open Sans, sans-serif"),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Open Sans, sans-serif"
                )
            )
            
            render_time = time.time() - start_time
            logger.info(f"Rendered teacher workload plot in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting teacher workload: {e}")
            return None
    
    def plot_room_usage(self, room_usage, metric='Nutzungshäufigkeit', top_n=10):
        """Erstellt ein Balkendiagramm für die Raumnutzung"""
        try:
            if room_usage.empty:
                return None
            
            start_time = time.time()
            
            # Metriklabels für die Anzeige
            metric_labels = {
                'Nutzungshäufigkeit': 'Häufigkeit der Nutzung',
                'Auslastung_Prozent': 'Auslastung in Prozent',
                'Durchschnitt_pro_Tag': 'Durchschnittliche Nutzung pro Tag'
            }
            
            # Sortieren nach gewählter Metrik
            sorted_data = room_usage.sort_values(metric, ascending=False).head(top_n)
            
            # Diagramm erstellen
            fig = px.bar(
                sorted_data,
                x='Raum',
                y=metric,
                title=f'Top {top_n} Räume nach {metric_labels.get(metric, metric)}',
                labels={metric: metric_labels.get(metric, metric), 'Raum': 'Raumnummer'},
                color=metric,
                color_continuous_scale='Viridis'
            )
            
            # Layout anpassen
            fig.update_layout(
                xaxis_tickangle=-45,
                font=dict(family="Open Sans, sans-serif"),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Open Sans, sans-serif"
                )
            )
            
            render_time = time.time() - start_time
            logger.info(f"Rendered room usage plot in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error plotting room usage: {e}")
            return None
    
    def create_student_timetable_view(self, student_timetable):
        """Erstellt eine visuelle Darstellung des Stundenplans eines Schülers"""
        try:
            if student_timetable.empty:
                return None
            
            start_time = time.time()
            
            # Pivot-Tabelle für den Stundenplan erstellen
            pivot = student_timetable.pivot_table(
                index='Stunde', 
                columns='Tag', 
                values='Fach', 
                aggfunc='first'
            ).fillna('')
            
            # Wochentage in richtige Reihenfolge bringen
            wochentage = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag']
            pivot = pivot.reindex(columns=wochentage)
            
            # Farbkodierung für Fächer erstellen
            unique_subjects = [subj for subj in student_timetable['Fach'].unique() if subj]
            subject_colors = {}
            
            # Farbpalette für Fächer
            color_palette = [
                '#4299E1', '#805AD5', '#F56565', '#48BB78', '#ECC94B',
                '#ED8936', '#9F7AEA', '#38B2AC', '#667EEA', '#F687B3'
            ]
            
            for i, subject in enumerate(unique_subjects):
                subject_colors[subject] = color_palette[i % len(color_palette)]
            
            # Annotations für die Zellen erstellen
            annotations = []
            
            for i, stunde in enumerate(pivot.index):
                for j, tag in enumerate(pivot.columns):
                    fach = pivot.loc[stunde, tag]
                    if fach:
                        # Informationen zum Fach aus dem Originaldatensatz abrufen
                        fach_info = student_timetable[
                            (student_timetable['Tag'] == tag) & 
                            (student_timetable['Stunde'] == stunde) & 
                            (student_timetable['Fach'] == fach)
                        ]
                        
                        if not fach_info.empty:
                            lehrer = fach_info.iloc[0]['Lehrer']
                            raum = fach_info.iloc[0]['Raum']
                            
                            text = f"{fach}<br>{lehrer}<br>{raum}"
                        else:
                            text = fach
                        
                        annotations.append(dict(
                            x=j,
                            y=i,
                            text=text,
                            showarrow=False,
                            font=dict(
                                color='white' if fach else 'black',
                                size=10
                            )
                        ))
            
            # Heatmap erstellen mit Farbkodierung nach Fach
            z = []
            hover_texts = []
            
            for i, stunde in enumerate(pivot.index):
                z_row = []
                hover_row = []
                
                for j, tag in enumerate(pivot.columns):
                    fach = pivot.loc[stunde, tag]
                    
                    # Zahlenwert für die Heatmap (für Farbkodierung)
                    if fach:
                        z_row.append(list(unique_subjects).index(fach) + 1)
                        
                        # Informationen für den Hover-Text
                        fach_info = student_timetable[
                            (student_timetable['Tag'] == tag) & 
                            (student_timetable['Stunde'] == stunde) & 
                            (student_timetable['Fach'] == fach)
                        ]
                        
                        if not fach_info.empty:
                            lehrer = fach_info.iloc[0]['Lehrer']
                            raum = fach_info.iloc[0]['Raum']
                            hover_row.append(f"Fach: {fach}<br>Lehrer: {lehrer}<br>Raum: {raum}")
                        else:
                            hover_row.append(f"Fach: {fach}")
                    else:
                        z_row.append(0)
                        hover_row.append("Keine Stunde")
                
                z.append(z_row)
                hover_texts.append(hover_row)
            
            # Colorscale basierend auf den Fächern erstellen
            colorscale = [[0, '#F3F4F6']]  # Hellgrau für leere Zellen
            
            for i, subject in enumerate(unique_subjects):
                colorscale.append([(i + 1) / (len(unique_subjects) + 1), subject_colors[subject]])
                colorscale.append([(i + 1.9) / (len(unique_subjects) + 1), subject_colors[subject]])
            
            # Heatmap erstellen
            fig = go.Figure(data=go.Heatmap(
                z=z,
                x=pivot.columns,
                y=pivot.index,
                colorscale=colorscale,
                showscale=False,
                text=hover_texts,
                hoverinfo='text'
            ))
            
            # Layout anpassen
            fig.update_layout(
                title=f"Stundenplan",
                xaxis=dict(side='top', title=''),
                yaxis=dict(title='Stunde', dtick=1),
                height=500,
                margin=dict(l=40, r=40, t=80, b=40),
                font=dict(family="Open Sans, sans-serif"),
                annotations=annotations
            )
            
            render_time = time.time() - start_time
            logger.info(f"Rendered student timetable view in {render_time:.2f} seconds")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating student timetable view: {e}")
            return None

# Styling und UI-Komponenten
def apply_custom_css():
    """Wendet benutzerdefiniertes CSS für die Anwendung an"""
    st.markdown("""
    <style>
        /* Basisstile */
        :root {
            --primary-color: #2563EB;
            --secondary-color: #7C3AED;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --danger-color: #EF4444;
            --info-color: #3B82F6;
            --light-color: #F3F4F6;
            --dark-color: #1F2937;
            --white: #FFFFFF;
            --black: #000000;
            --background-color: #F9FAFB;
            --text-color: #111827;
            --muted-color: #6B7280;
            --border-radius: 0.5rem;
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Globale Stile */
        body {
            font-family: var(--font-family);
            color: var(--text-color);
            background-color: var(--background-color);
        }
        
        /* Überschriften */
        .main-header {
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        
        .sub-header {
            font-size: clamp(1.4rem, 2.5vw, 1.8rem);
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
            line-height: 1.3;
        }
        
        .section-header {
            font-size: clamp(1.2rem, 2vw, 1.5rem);
            font-weight: 600;
            color: var(--dark-color);
            margin: 1rem 0 0.5rem 0;
            padding-bottom: 0.25rem;
            border-bottom: 2px solid var(--primary-color);
        }
        
        /* Karten und Container */
        .card {
            background-color: var(--white);
            border-radius: var(--border-radius);
            padding: clamp(0.8rem, 2vw, 1.2rem);
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary-color);
            box-shadow: var(--box-shadow);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        
        .card-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .card-content {
            padding: 0.5rem 0;
        }
        
        .card-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.75rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--light-color);
            font-size: 0.875rem;
            color: var(--muted-color);
        }
        
        .info-card {
            background-color: #EFF6FF;
            border-left: 4px solid var(--info-color);
        }
        
        .warning-card {
            background-color: #FEF3C7;
            border-left: 4px solid var(--warning-color);
        }
        
        .success-card {
            background-color: #ECFDF5;
            border-left: 4px solid var(--success-color);
        }
        
        .danger-card {
            background-color: #FEE2E2;
            border-left: 4px solid var(--danger-color);
        }
        
        .dashboard-tile {
            background-color: var(--white);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            box-shadow: var(--box-shadow);
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .dashboard-tile-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .dashboard-tile-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        /* Text-Stile */
        .info-text {
            font-size: clamp(0.9rem, 1.5vw, 1rem);
            color: var(--muted-color);
            line-height: 1.5;
        }
        
        .highlight {
            color: var(--primary-color);
            font-weight: 600;
        }
        
        .small-text {
            font-size: 0.875rem;
            color: var(--muted-color);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 0.25rem;
        }
        
        .stat-label {
            font-size: 0.875rem;
            color: var(--muted-color);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Tabellen und Daten */
        .dataframe {
            font-size: clamp(0.75rem, 1.2vw, 0.875rem);
            width: 100%;
            border-collapse: collapse;
        }
        
        .dataframe th {
            background-color: #EFF6FF;
            padding: 0.5rem 0.75rem;
            text-align: left;
            font-weight: 600;
            color: var(--dark-color);
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .dataframe td {
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid var(--light-color);
        }
        
        .dataframe tr:hover {
            background-color: var(--light-color);
        }
        
        /* Buttons und Interaktive Elemente */
        .stButton>button {
            background-color: var(--primary-color);
            color: var(--white);
            border-radius: 0.375rem;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .stButton>button:hover {
            background-color: #1D4ED8;
        }
        
        .stButton>button:active {
            background-color: #1E40AF;
        }
        
        .stSelectbox [data-baseweb=select] {
            border-radius: 0.375rem;
        }
        
        .stTextInput [data-baseweb=input] {
            border-radius: 0.375rem;
        }
        
        /* Sidebar-Anpassungen */
        .css-1d391kg {
            padding-top: 2rem;
        }
        
        .sidebar .sidebar-content {
            background-color: var(--white);
        }
        
        /* Tabs-Anpassungen */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
            background-color: var(--light-color);
            border-radius: 0.5rem;
            padding: 0.25rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 2.5rem;
            white-space: pre-wrap;
            border-radius: 0.375rem;
            color: var(--dark-color);
            background-color: transparent;
            border: none;
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--white);
            color: var(--primary-color);
            font-weight: 600;
            box-shadow: var(--box-shadow);
        }
        
        /* Drag-and-Drop Bereich */
        .dropzone {
            border: 2px dashed var(--primary-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            background-color: #EFF6FF;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .dropzone:hover {
            background-color: #DBEAFE;
        }
        
        /* Expander-Anpassungen */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: var(--dark-color);
        }
        
        .streamlit-expanderContent {
            padding: 0.75rem;
            border-radius: 0 0 var(--border-radius) var(--border-radius);
            background-color: var(--white);
        }
        
        /* Responsive Anpassungen */
        @media (max-width: 768px) {
            .stTabs [data-baseweb="tab-list"] {
                flex-wrap: wrap;
            }
            
            .stTabs [data-baseweb="tab"] {
                font-size: 0.8rem;
                padding: 0.3rem 0.5rem;
            }
            
            .card {
                padding: 0.8rem;
            }
            
            .dataframe {
                font-size: 0.7rem;
            }
            
            .main-header {
                font-size: 1.5rem;
            }
            
            .sub-header {
                font-size: 1.2rem;
            }
        }
        
        /* Animations und Übergänge */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .scale-in {
            animation: scaleIn 0.3s ease-out;
        }
        
        @keyframes scaleIn {
            from { transform: scale(0.95); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--light-color);
        }
        
        ::-webkit-scrollbar-thumb {
            background: #94A3B8;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #64748B;
        }
        
        /* Print-Optimierungen */
        @media print {
            .stButton, .stSidebar, .stCheckbox, .stSelectbox, .stFileUploader {
                display: none !important;
            }
            
            .main .block-container {
                max-width: 100% !important;
                padding: 0 !important;
            }
            
            h1, h2, h3, h4, h5, h6 {
                break-after: avoid;
            }
            
            table, figure {
                break-inside: avoid;
            }
        }
        
        /* Zusätzliche Komponenten-Stile */
        .metric-container {
            background-color: var(--white);
            border-radius: var(--border-radius);
            padding: 1rem;
            box-shadow: var(--box-shadow);
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.25rem;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: var(--muted-color);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background-color: var(--dark-color);
            color: var(--white);
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """, unsafe_allow_html=True)

def create_custom_metric(label, value, delta=None, delta_color="normal", help_text=None):
    """Erstellt eine benutzerdefinierte Metrik-Anzeige"""
    if delta:
        delta_sign = "+" if delta > 0 else ""
        delta_html = f"""
        <div style="color: {'#10B981' if delta_color == 'normal' and delta > 0 or delta_color == 'inverse' and delta < 0 else '#EF4444'}; font-size: 0.8rem; display: flex; align-items: center; margin-top: 0.25rem;">
            {delta_sign}{delta}% {' ▲' if delta > 0 else ' ▼'}
        </div>
        """
    else:
        delta_html = ""
    
    help_icon = ""
    if help_text:
        help_icon = f"""
        <div class="tooltip" style="margin-left: 0.25rem;">
            ℹ️
            <span class="tooltip-text">{help_text}</span>
        </div>
        """
    
    html = f"""
    <div class="metric-container scale-in">
        <div class="metric-value">{value}</div>
        {delta_html}
        <div class="metric-label">
            {label} {help_icon}
        </div>
    </div>
    """
    
    return st.markdown(html, unsafe_allow_html=True)

def create_info_card(title, content, card_type="info"):
    """Erstellt eine Infokarte mit verschiedenen Stilen"""
    card_class = f"card {card_type}-card"
    
    html = f"""
    <div class="{card_class} scale-in">
        <div class="card-header">{title}</div>
        <div class="card-content">{content}</div>
    </div>
    """
    
    return st.markdown(html, unsafe_allow_html=True)

def create_dashboard_tile(title, content, footer=None, icon=None):
    """Erstellt eine Dashboard-Kachel"""
    icon_html = f'<span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''
    footer_html = f'<div class="card-footer">{footer}</div>' if footer else ''
    
    html = f"""
    <div class="dashboard-tile scale-in">
        <div class="dashboard-tile-header">{icon_html}{title}</div>
        <div class="dashboard-tile-content">{content}</div>
        {footer_html}
    </div>
    """
    
    return st.markdown(html, unsafe_allow_html=True)

def get_lottie_animation(animation_name):
    """Lädt eine Lottie-Animation für UI-Elemente"""
    animations = {
        "loading": "https://assets5.lottiefiles.com/packages/lf20_usmfx6bp.json",
        "success": "https://assets8.lottiefiles.com/private_files/lf30_t26law.json",
        "error": "https://assets2.lottiefiles.com/packages/lf20_qpwbiyxf.json",
        "empty": "https://assets6.lottiefiles.com/packages/lf20_dmw3p65n.json",
        "data": "https://assets10.lottiefiles.com/packages/lf20_xlkxtmul.json",
        "chart": "https://assets2.lottiefiles.com/private_files/lf30_ajzyv3qf.json"
    }
    
    try:
        if animation_name in animations:
            return requests.get(animations[animation_name]).json()
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading Lottie animation: {e}")
        return None

def render_lottie_animation(animation_data, height=200, width=None):
    """Rendert eine Lottie-Animation"""
    if animation_data:
        st_lottie(animation_data, height=height, width=width, key=f"lottie_{uuid.uuid4().hex[:8]}")

def render_dropzone_area(accept_files=None, max_file_size=200, multiple_files=True, callback=None):
    """Rendert eine Drag-and-Drop-Zone für Datei-Uploads"""
    if accept_files is None:
        accept_files = ['.json', '.csv', '.xlsx']
    
    drop_id = f"dropzone_{uuid.uuid4().hex[:8]}"
    
    drop_str = create_dropzone(
        id=drop_id,
        label="Dateien hierher ziehen und ablegen oder klicken zum Auswählen",
        max_file_size=max_file_size,
        accepted_files=",".join(accept_files),
        multiple=multiple_files,
    )
    
    if drop_str:
        uploaded_files = json.loads(drop_str)
        if callback and callable(callback):
            return callback(uploaded_files)
        return uploaded_files
    return None

def create_menu(items, default_index=0, orientation="horizontal", key=None):
    """Erstellt ein benutzerdefiniertes Menü"""
    if orientation == "horizontal":
        menu = option_menu(
            menu_title=None,
            options=items,
            default_index=default_index,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "var(--primary-color)", "font-size": "1rem"},
                "nav-link": {
                    "font-size": "0.9rem",
                    "text-align": "center",
                    "margin": "0px",
                    "padding": "0.5rem 1rem",
                    "border-radius": "0.5rem",
                },
                "nav-link-selected": {"background-color": "var(--primary-color)", "color": "white"},
            },
            key=key
        )
    else:
        menu = option_menu(
            menu_title="Menü",
            options=items,
            default_index=default_index,
            styles={
                "container": {"padding": "1rem", "background-color": "#F3F4F6"},
                "icon": {"color": "var(--primary-color)", "font-size": "1rem"},
                "nav-link": {"font-size": "0.9rem", "padding": "0.5rem 1rem"},
                "nav-link-selected": {"background-color": "var(--primary-color)"},
            },
            key=key
        )
    
    return menu

def display_dataframe_with_download(df, name="data", key=None):
    """Zeigt ein DataFrame an mit Möglichkeit zum Download"""
    if df.empty:
        st.info("Keine Daten verfügbar.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.download_button(
            label="CSV herunterladen",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"{name}_{datetime.now().strftime('%Y-%m-%d')}.csv",
            mime='text/csv',
            key=f"download_csv_{key or uuid.uuid4().hex[:8]}"
        )
        
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False, engine='xlsxwriter')
        excel_buffer.seek(0)
        
        st.download_button(
            label="Excel herunterladen",
            data=excel_buffer,
            file_name=f"{name}_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
            mime='application/vnd.ms-excel',
            key=f"download_excel_{key or uuid.uuid4().hex[:8]}"
        )

def display_file_uploader(allowed_types=None, label=None, help_text=None, key=None):
    """Erweiterte Datei-Upload-Komponente"""
    if allowed_types is None:
        allowed_types = ["json", "csv", "xlsx"]
    
    label = label or "Datei hochladen"
    help_text = help_text or f"Unterstützte Dateitypen: {', '.join(allowed_types)}"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(label, type=allowed_types, help=help_text, key=key)
    
    with col2:
        if st.button("Beispieldaten laden", key=f"load_sample_{key or uuid.uuid4().hex[:8]}"):
            # Beispieldaten sind je nach Dateityp unterschiedlich
            if "json" in allowed_types:
                return "sample_data.json"  # Hier wird nur der Pfad zurückgegeben
            elif "csv" in allowed_types:
                return "sample_data.csv"
    
    return uploaded_file

def create_tabs_with_icons(tabs_data):
    """Erstellt Tabs mit Icons"""
    tabs = st.tabs([f"{data['icon']} {data['label']}" for data in tabs_data])
    return tabs

def create_expander_with_content(title, content_func, expanded=False, icon=None):
    """Erstellt einen Expander mit dynamischem Inhalt"""
    icon_str = f"{icon} " if icon else ""
    with st.expander(f"{icon_str}{title}", expanded=expanded):
        content_func()

def toggle_with_action(label, key=None, default=False, on_change=None):
    """Erstellt einen erweiterten Toggle-Switch mit Aktion"""
    key = key or f"toggle_{uuid.uuid4().hex[:8]}"
    state = tog.st_toggle_switch(label, key=key, default_value=default)
    
    if on_change and callable(on_change):
        if state:
            on_change()
    
    return state

def get_app_download_link(file_name="app.py", label="App herunterladen"):
    """Generiert einen Link zum Herunterladen der App"""
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = file.read()
        
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">{label}</a>'
        return href
    except Exception as e:
        logger.error(f"Error creating app download link: {e}")
        return None

# Cache-Management und Datenpersistenz
@st.cache_data(ttl=3600)
def get_cache_version():
    """Generiert eine Cache-Version für die aktuelle Sitzung"""
    return int(time.time())

def reset_cache():
    """Setzt den Cache zurück"""
    global cache, cache_version
    cache = {}
    cache_version = get_cache_version()
    st.cache_data.clear()
    
def generate_unique_key(prefix=""):
    """Generiert einen eindeutigen Schlüssel für Widget-IDs"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# Erweiterte Cache-Funktionen
def get_cached_data(key, ttl=CACHE_TTL):
    """Lädt Daten aus dem Cache mit TTL (Time To Live)"""
    global cache
    
    if key in cache:
        timestamp, data = cache[key]
        if time.time() - timestamp < ttl:
            return data
    
    return None

def set_cached_data(key, data):
    """Speichert Daten im Cache mit aktuellem Zeitstempel"""
    global cache
    cache[key] = (time.time(), data)
    return True

def clear_expired_cache(ttl=CACHE_TTL):
    """Löscht abgelaufene Cache-Einträge"""
    global cache
    current_time = time.time()
    expired_keys = [k for k, (timestamp, _) in cache.items() if current_time - timestamp > ttl]
    
    for key in expired_keys:
        del cache[key]
    
    return len(expired_keys)

def get_cache_stats():
    """Gibt Statistiken zum Cache zurück"""
    global cache
    total_size = sum(sys.getsizeof(data) for _, data in cache.values())
    return {
        "entries": len(cache),
        "size_bytes": total_size,
        "size_mb": total_size / (1024 * 1024)
    }

# Datenversionsmanagement
class DataVersionManager:
    """Verwaltet Versionen von importierten Daten"""
    
    def __init__(self, base_dir="data_versions"):
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    def save_version(self, data, name, metadata=None):
        """Speichert eine neue Datenversion"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{name}_{timestamp}"
        version_dir = os.path.join(self.base_dir, version_id)
        
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
        
        # Daten speichern
        data_path = os.path.join(version_dir, "data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        
        # Metadaten speichern
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "version_id": version_id,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat()
        })
        
        meta_path = os.path.join(version_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        
        return version_id
    
    def load_version(self, version_id):
        """Lädt eine gespeicherte Datenversion"""
        version_dir = os.path.join(self.base_dir, version_id)
        
        if not os.path.exists(version_dir):
            return None, None
        
        # Daten laden
        data_path = os.path.join(version_dir, "data.pkl")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
        # Metadaten laden
        meta_path = os.path.join(version_dir, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return data, metadata
    
    def list_versions(self, name=None):
        """Listet alle verfügbaren Versionen auf"""
        versions = []
        
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            
            if os.path.isdir(item_path):
                if name is None or item.startswith(name):
                    meta_path = os.path.join(item_path, "metadata.json")
                    
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        
                        versions.append(metadata)
        
        # Nach Zeitstempel sortieren (neueste zuerst)
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return versions
    
    def delete_version(self, version_id):
        """Löscht eine Datenversion"""
        version_dir = os.path.join(self.base_dir, version_id)
        
        if not os.path.exists(version_dir):
            return False
        
        # Rekursiv Verzeichnis löschen
        for root, dirs, files in os.walk(version_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        
        os.rmdir(version_dir)
        return True

# Dropzone-Funktionalität für Datei-Uploads
def create_dropzone(id="dropzone", label="Dateien hier ablegen", accepted_files=".csv,.json,.xlsx", multiple=True, max_file_size=200, height="200px", key=None):
    """
    Erstellt eine Drag-and-Drop-Zone für Datei-Uploads.
    
    Args:
        id (str): Die ID für das Dropzone-Element
        label (str): Der angezeigte Text in der Dropzone
        accepted_files (str): Kommagetrennte Liste akzeptierter Dateitypen
        multiple (bool): Ob mehrere Dateien hochgeladen werden können
        max_file_size (int): Maximale Dateigröße in MB
        height (str): Höhe der Dropzone (CSS-Wert)
        key (str): Eindeutiger Schlüssel für das Streamlit-Element
        
    Returns:
        str: JSON-String mit Informationen über hochgeladene Dateien oder None
    """
    if key is None:
        key = f"dropzone_{id}"
    
    # JavaScript für die Dropzone
    dropzone_js = f"""
    <script src="https://unpkg.com/dropzone@5/dist/min/dropzone.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/dropzone@5/dist/min/dropzone.min.css" type="text/css" />
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Dropzone konfigurieren
            Dropzone.autoDiscover = false;
            
            var dropzoneOptions = {{
                url: "#",
                autoProcessQueue: false,
                uploadMultiple: {'true' if multiple else 'false'},
                parallelUploads: 10,
                maxFilesize: {max_file_size},
                acceptedFiles: "{accepted_files}",
                dictDefaultMessage: "{label}",
                clickable: true,
                createImageThumbnails: true,
                maxFiles: {'null' if multiple else '1'},
                addRemoveLinks: true,
                dictRemoveFile: "Entfernen",
                dictCancelUpload: "Abbrechen",
                dictCancelUploadConfirmation: "Upload wirklich abbrechen?",
                dictFileTooBig: "Datei ist zu groß ({{filesize}}MB). Maximale Dateigröße: {{maxFilesize}}MB.",
                dictInvalidFileType: "Dieser Dateityp wird nicht unterstützt.",
                dictMaxFilesExceeded: "Maximale Anzahl an Dateien überschritten.",
                init: function() {{
                    var myDropzone = this;
                    
                    // Bei Hinzufügen einer Datei
                    this.on("addedfile", function(file) {{
                        console.log("Datei hinzugefügt:", file.name);
                        
                        // Datei in Base64 konvertieren
                        var reader = new FileReader();
                        reader.readAsDataURL(file);
                        reader.onload = function() {{
                            file.dataURL = reader.result;
                            updateStreamlitState();
                        }};
                    }});
                    
                    // Bei Entfernen einer Datei
                    this.on("removedfile", function(file) {{
                        console.log("Datei entfernt:", file.name);
                        updateStreamlitState();
                    }});
                    
                    // Streamlit-Status aktualisieren
                    function updateStreamlitState() {{
                        var files = [];
                        myDropzone.files.forEach(function(file) {{
                            if (file.dataURL) {{
                                files.push({{
                                    name: file.name,
                                    type: file.type,
                                    size: file.size,
                                    content: file.dataURL
                                }});
                            }}
                        }});
                        
                        // An Streamlit senden
                        if (files.length > 0) {{
                            const jsonStr = JSON.stringify(files);
                            window.parent.postMessage({{
                                type: "streamlit:setComponentValue",
                                value: jsonStr,
                                dataType: "json"
                            }}, "*");
                        }} else {{
                            window.parent.postMessage({{
                                type: "streamlit:setComponentValue",
                                value: "",
                                dataType: "json"
                            }}, "*");
                        }}
                    }}
                }}
            }};
            
            // Dropzone erstellen
            new Dropzone("#{id}", dropzoneOptions);
        }});
    </script>
    """
    
    # HTML für die Dropzone
    dropzone_html = f"""
    <div class="file-drop-container" style="margin-bottom: 1rem;">
        <form id="{id}" class="dropzone" style="height: {height}; min-height: 100px; border: 2px dashed #2563EB; border-radius: 0.5rem; background-color: #EFF6FF;">
            <div class="dz-message" style="margin: 1rem 0;">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: #2563EB;">
                    <span class="material-icons">cloud_upload</span>
                </div>
                <div style="font-weight: 500;">{label}</div>
                <div style="font-size: 0.875rem; color: #6B7280; margin-top: 0.5rem;">
                    Max. Dateigröße: {max_file_size} MB
                </div>
            </div>
        </form>
    </div>
    """
    
    # Komponente in Streamlit rendern
    component_value = st.components.v1.html(
        dropzone_html + dropzone_js,
        height=int(height.replace("px", "")) + 50 if "px" in height else 250,
        key=key
    )
    
    return component_value

# Drag-and-Drop Datei-Upload
def render_dropzone(accept_files=None, max_file_size=200, multiple_files=True, on_upload=None):
    """Rendert eine erweiterte Drag-and-Drop-Zone für Datei-Uploads"""
    if accept_files is None:
        accept_files = ['.json', '.csv', '.xlsx']
    
    # Eindeutige ID für die Dropzone
    drop_id = generate_unique_key("drop")
    
    # CSS für die Dropzone
    st.markdown("""
    <style>
    .file-drop-area {
        border: 2px dashed #2563EB;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        transition: background-color 0.3s;
        background-color: #EFF6FF;
        margin-bottom: 1rem;
    }
    .file-drop-area:hover {
        background-color: #DBEAFE;
    }
    .file-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: #2563EB;
    }
    .file-drop-message {
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .file-drop-info {
        font-size: 0.875rem;
        color: #6B7280;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Erstelle die Dropzone
    drop_str = create_dropzone(
        id=drop_id,
        label="Dateien hierher ziehen und ablegen oder klicken zum Auswählen",
        max_file_size=max_file_size,
        accepted_files=",".join(accept_files),
        multiple=multiple_files,
        height="200px",
        key=f"dropzone_{uuid.uuid4().hex[:8]}"
    )
    
    # Wenn Dateien hochgeladen wurden
    if drop_str:
        try:
            uploaded_files = json.loads(drop_str)
            
            # Verarbeite Dateien
            if on_upload and callable(on_upload):
                return on_upload(uploaded_files)
            
            processed_files = []
            
            for file_info in uploaded_files:
                file_name = file_info.get("name", "")
                file_type = file_name.split(".")[-1].lower() if "." in file_name else ""
                file_content = file_info.get("content", "")
                
                # Base64-decodieren
                if file_content:
                    try:
                        file_content = base64.b64decode(file_content.split(",")[1])
                        
                        # Dateityp-spezifische Verarbeitung
                        if file_type == "json":
                            data = json.loads(file_content.decode("utf-8"))
                        elif file_type == "csv":
                            data = pd.read_csv(io.BytesIO(file_content))
                        elif file_type in ["xlsx", "xls"]:
                            data = pd.read_excel(io.BytesIO(file_content))
                        else:
                            data = file_content
                        
                        processed_files.append({
                            "name": file_name,
                            "type": file_type,
                            "size": len(file_content),
                            "data": data
                        })
                    except Exception as e:
                        st.error(f"Fehler bei der Verarbeitung von {file_name}: {e}")
            
            return processed_files
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten der hochgeladenen Dateien: {e}")
            return None
    
    return None

# Datei-Import-Funktionen mit verbesserten Fehlermeldungen und Validierung
@st.cache_data(ttl=3600)
def load_json_file(file_path):
    """JSON-Datei laden mit Caching, Fehlerbehandlung und Validierung"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Prüfe auf leere Daten
            if not data:
                st.warning(f"Die Datei {file_path} enthält keine Daten.")
                return None
                
            return data
    except FileNotFoundError:
        st.error(f"Die Datei {file_path} wurde nicht gefunden.")
        return None
    except json.JSONDecodeError:
        st.error(f"Die Datei {file_path} enthält ungültiges JSON.")
        return None
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei {file_path}: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def convert_timetable_to_df(timetable_data):
    """Konvertiert die Stundenplan-JSON-Daten in ein DataFrame"""
    try:
        # Wenn die Daten bereits als DataFrame vorliegen
        if isinstance(timetable_data, pd.DataFrame):
            return timetable_data
            
        rows = []
        for entry in timetable_data:
            tag = entry.get('tag', '')
            stunde = entry.get('stunde', 0)
            faecher = entry.get('faecher', [])
            
            if faecher:
                for fach_entry in faecher:
                    rows.append({
                        'Tag': tag,
                        'Stunde': stunde,
                        'Fach': fach_entry.get('fach', ''),
                        'Lehrer': fach_entry.get('lehrer', ''),
                        'Raum': fach_entry.get('raum', '')
                    })
            else:
                # Leerer Zeitslot
                rows.append({
                    'Tag': tag,
                    'Stunde': stunde,
                    'Fach': '',
                    'Lehrer': '',
                    'Raum': ''
                })
        
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"Fehler bei der Konvertierung der Stundenplan-Daten: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def convert_courses_to_df(courses_data):
    """Konvertiert die Kurs-JSON-Daten in ein DataFrame"""
    try:
        # Wenn die Daten bereits als DataFrame vorliegen
        if isinstance(courses_data, pd.DataFrame):
            return courses_data
            
        rows = []
        for course in courses_data:
            if not isinstance(course, dict):
                continue
                
            rows.append({
                'Kurs': course.get('kurs', ''),
                'Kurs_ID': course.get('kurs_id', 0),
                'Kurstyp': course.get('kurs_type', ''),
                'Teilnehmerzahl': course.get('participants_count', 0),
                'Teilnehmer': course.get('participants', [])
            })
        
        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        st.error(f"Fehler bei der Konvertierung der Kurs-Daten: {str(e)}")
        return pd.DataFrame()

# Funktion zum Laden einer JSON-Datei mit erweiterter Validierung
@st.cache_data(ttl=3600)
def load_json_with_validation(file_path):
    """Lädt und validiert JSON-Dateien mit Caching"""
    try:
        # Cache-Key generieren
        cache_key = f"json_{file_path}_{cache_version}"
        
        # Prüfe, ob Daten bereits im Cache sind
        cached_data = get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Daten aus Datei laden
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Prüfe auf leere Daten
        if not data:
            logger.warning(f"Leere Daten in {file_path}")
            st.warning(f"Die Datei {file_path} enthält keine Daten.")
            return None
        
        # Validiere Daten (hier können spezifische Prüfungen hinzugefügt werden)
        if file_path.endswith('timetable.json'):
            # Validiere Stundenplan-Daten
            if not isinstance(data, list):
                logger.error(f"Ungültiges Format in {file_path}: Erwartet wurde eine Liste")
                st.error(f"Ungültiges Format in {file_path}: Erwartet wurde eine Liste")
                return None
        
        # In den Cache speichern
        set_cached_data(cache_key, data)
        return data
    except FileNotFoundError:
        logger.error(f"Datei nicht gefunden: {file_path}")
        st.error(f"Datei nicht gefunden: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Ungültiges JSON-Format in {file_path}: {e}")
        st.error(f"Ungültiges JSON-Format in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Fehler beim Laden der Datei {file_path}: {e}")
        st.error(f"Fehler beim Laden der Datei {file_path}: {e}")
        return None

def save_uploaded_file(uploaded_file, target_dir="json", versioning=True):
    """Speichert eine hochgeladene Datei mit optionaler Versionierung"""
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Pfad für die Datei
        file_path = os.path.join(target_dir, uploaded_file.name)
        
        # Bei Versionierung: Erstelle Version falls die Datei bereits existiert
        if versioning and os.path.exists(file_path):
            # Erstelle Timestamp für Versionierung
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Erstelle Verzeichnis für Versionen
            versions_dir = os.path.join(target_dir, "versions")
            if not os.path.exists(versions_dir):
                os.makedirs(versions_dir)
            
            # Kopiere bestehende Datei in Versionen-Verzeichnis
            filename, ext = os.path.splitext(uploaded_file.name)
            version_path = os.path.join(versions_dir, f"{filename}_{timestamp}{ext}")
            
            with open(file_path, "rb") as src, open(version_path, "wb") as dst:
                dst.write(src.read())
            
            logger.info(f"Bestehende Datei als Version gespeichert: {version_path}")
        
        # Speichere neue Datei
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Datei gespeichert: {file_path}")
        return file_path, True
    except Exception as e:
        logger.error(f"Fehler beim Speichern der Datei: {e}")
        st.error(f"Fehler beim Speichern der Datei: {e}")
        return None, False

# Datenbankanbindung und Persistenz
class DatabaseManager:
    """Erweiterte Klasse für Datenbankoperationen mit Unterstützung für MongoDB und SQLite"""
    
    def __init__(self, db_type="sqlite", connection_string=None):
        self.db_type = db_type.lower()
        self.connection_string = connection_string
        self.connection = None
        self.is_connected = False
        
        # Standardwerte für Connection-Strings
        if self.connection_string is None:
            if self.db_type == "sqlite":
                self.connection_string = DEFAULT_DB_PATH
            elif self.db_type == "mongodb":
                self.connection_string = MONGODB_URI
    
    def connect(self):
        """Stellt eine Verbindung zur Datenbank her mit erweiterten Fehlerprüfungen"""
        try:
            if self.db_type == "sqlite":
                # Stelle sicher, dass das Verzeichnis existiert
                db_dir = os.path.dirname(self.connection_string)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir)
                
                # Öffne Verbindung
                self.connection = sqlite3.connect(self.connection_string)
                
                # Aktiviere Foreign-Key-Constraints
                self.connection.execute("PRAGMA foreign_keys = ON")
                
                # Aktiviere WAL-Modus für bessere Performance
                self.connection.execute("PRAGMA journal_mode = WAL")
                
                self.is_connected = True
                logger.info(f"Verbunden mit SQLite: {self.connection_string}")
                return True
            
            elif self.db_type == "mongodb":
                # Verbindung mit Timeout
                self.connection = pymongo.MongoClient(
                    self.connection_string,
                    serverSelectionTimeoutMS=5000  # 5 Sekunden Timeout
                )
                
                # Teste Verbindung
                self.connection.admin.command('ping')
                
                self.is_connected = True
                logger.info(f"Verbunden mit MongoDB: {self.connection_string}")
                return True
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                st.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"SQLite-Fehler: {e}")
            st.error(f"SQLite-Fehler: {e}")
            return False
        
        except pymongo.errors.ServerSelectionTimeoutError as e:
            logger.error(f"MongoDB-Verbindungsfehler: {e}")
            st.error(f"MongoDB-Verbindungsfehler: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Datenbankfehler: {e}")
            st.error(f"Datenbankfehler: {e}")
            return False
    
    def disconnect(self):
        """Trennt die Verbindung zur Datenbank"""
        try:
            if self.is_connected:
                if self.db_type == "sqlite":
                    self.connection.close()
                elif self.db_type == "mongodb":
                    self.connection.close()
                
                self.is_connected = False
                logger.info("Datenbankverbindung getrennt")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Fehler beim Trennen der Datenbankverbindung: {e}")
            return False
    
    def save_dataframe(self, df, table_name, if_exists="replace", index=False):
        """Speichert ein DataFrame in der Datenbank mit erweiterten Optionen"""
        try:
            if not self.is_connected:
                if not self.connect():
                    return False
            
            if self.db_type == "sqlite":
                # DataFrame in SQLite speichern
                df.to_sql(table_name, self.connection, if_exists=if_exists, index=index)
                
                # Größe der Tabelle abrufen
                cursor = self.connection.cursor()
                cursor.execute(f"SELECT count(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                logger.info(f"{count} Datensätze in SQLite-Tabelle {table_name} gespeichert")
                return True
            
            elif self.db_type == "mongodb":
                # MongoDB-Datenbank und Collection auswählen
                db = self.connection["kursplan_analyse"]
                collection = db[table_name]
                
                # DataFrame in Dictionary-Liste konvertieren
                records = df.to_dict("records")
                
                # Bestehende Daten löschen, wenn gewünscht
                if if_exists == "replace":
                    collection.delete_many({})
                
                # Daten einfügen
                if records:
                    # Batch-Insert für bessere Performance
                    if len(records) > 1000:
                        # In Batches von 1000 Dokumenten aufteilen
                        batch_size = 1000
                        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
                        
                        for batch in batches:
                            collection.insert_many(batch)
                    else:
                        collection.insert_many(records)
                
                logger.info(f"{len(records)} Datensätze in MongoDB-Collection {table_name} gespeichert")
                return True
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return False
        
        except Exception as e:
            logger.error(f"Fehler beim Speichern in der Datenbank: {e}")
            st.error(f"Fehler beim Speichern in der Datenbank: {e}")
            return False
    
    def load_dataframe(self, table_name, query=None, columns=None):
        """Lädt ein DataFrame aus der Datenbank mit erweiterten Filteroptionen"""
        try:
            if not self.is_connected:
                if not self.connect():
                    return pd.DataFrame()
            
            if self.db_type == "sqlite":
                # SQL-Abfrage erstellen
                sql = f"SELECT {', '.join(columns) if columns else '*'} FROM {table_name}"
                
                # WHERE-Klausel hinzufügen, wenn query vorhanden ist
                if query and isinstance(query, dict):
                    where_clauses = []
                    params = {}
                    
                    for key, value in query.items():
                        where_clauses.append(f"{key} = :{key}")
                        params[key] = value
                    
                    if where_clauses:
                        sql += " WHERE " + " AND ".join(where_clauses)
                    
                    return pd.read_sql_query(sql, self.connection, params=params)
                
                # Einfache Abfrage ohne Parameter
                return pd.read_sql_query(sql, self.connection)
            
            elif self.db_type == "mongodb":
                # MongoDB-Datenbank und Collection auswählen
                db = self.connection["kursplan_analyse"]
                collection = db[table_name]
                
                # Abfrage ausführen
                mongo_query = query or {}
                projection = {col: 1 for col in columns} if columns else None
                
                cursor = collection.find(mongo_query, projection)
                df = pd.DataFrame(list(cursor))
                
                # MongoDB-ID entfernen, falls vorhanden und nicht explizit angefordert
                if "_id" in df.columns and (columns is None or "_id" not in columns):
                    df = df.drop("_id", axis=1)
                
                return df
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Fehler beim Laden aus der Datenbank: {e}")
            st.error(f"Fehler beim Laden aus der Datenbank: {e}")
            return pd.DataFrame()
    
    def execute_query(self, query, params=None):
        """Führt eine benutzerdefinierte Abfrage aus"""
        try:
            if not self.is_connected:
                if not self.connect():
                    return None
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                self.connection.commit()
                
                # Versuche Ergebnisse abzurufen
                try:
                    return cursor.fetchall()
                except sqlite3.Error:
                    # Kein Ergebnis (z.B. bei INSERT, UPDATE, DELETE)
                    return cursor.rowcount
            
            elif self.db_type == "mongodb":
                # MongoDB unterstützt keine SQL-Abfragen
                # Hier könnte eine einfache Ausführung von MongoDB-Befehlen implementiert werden
                logger.warning("Benutzerdefinierte Abfragen werden für MongoDB nicht unterstützt")
                return None
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return None
        
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung der Abfrage: {e}")
            st.error(f"Fehler bei der Ausführung der Abfrage: {e}")
            return None
    
    def get_tables(self):
        """Gibt eine Liste aller Tabellen/Collections zurück"""
        try:
            if not self.is_connected:
                if not self.connect():
                    return []
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                return [row[0] for row in cursor.fetchall()]
            
            elif self.db_type == "mongodb":
                db = self.connection["kursplan_analyse"]
                return db.list_collection_names()
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return []
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Tabellen: {e}")
            return []

# Datenkonvertierungsfunktionen mit Parallelisierung
@st.cache_data(ttl=CACHE_TTL)
def convert_timetable_to_df(timetable_data):
    """Stundenplan-Daten in DataFrame konvertieren mit Parallelisierung für große Datensätze"""
    if not timetable_data:
        return pd.DataFrame()
    
    # Performance-Messung starten
    start_time = time.time()
    
    # Verwende Parallelisierung für große Datensätze
    if len(timetable_data) > 500:
        try:
            # Anzahl der CPU-Kerne für die Parallelisierung
            num_cores = min(os.cpu_count() or 4, 8)  # Maximal 8 Kerne verwenden
            
            # Daten in Chunks aufteilen
            chunk_size = len(timetable_data) // num_cores
            chunks = [timetable_data[i:i+chunk_size] for i in range(0, len(timetable_data), chunk_size)]
            
            # Parallele Verarbeitung mit ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                results = list(executor.map(_process_timetable_chunk, chunks))
            
            # Ergebnisse zusammenführen
            timetable_rows = []
            for result in results:
                timetable_rows.extend(result)
        except Exception as e:
            logger.error(f"Fehler bei der parallelen Verarbeitung: {e}")
            # Fallback zur sequentiellen Verarbeitung
            timetable_rows = _process_timetable_chunk(timetable_data)
    else:
        # Für kleinere Datensätze sequentielle Verarbeitung verwenden
        timetable_rows = _process_timetable_chunk(timetable_data)
    
    # DataFrame erstellen
    df = pd.DataFrame(timetable_rows)
    
    # Optimierung für bessere Filterung und Sortierung
    if not df.empty and 'Tag' in df.columns:
        # Wochentage in richtige Reihenfolge bringen
        weekday_order = {
            'Montag': 0, 
            'Dienstag': 1, 
            'Mittwoch': 2, 
            'Donnerstag': 3, 
            'Freitag': 4
        }
        
        df['Tag_Order'] = df['Tag'].map(weekday_order)
        df = df.sort_values(['Tag_Order', 'Stunde']).drop('Tag_Order', axis=1)
    
    # Performance-Messung beenden
    end_time = time.time()
    logger.info(f"Stundenplan-Konvertierung: {end_time - start_time:.2f} Sekunden")
    
    return df

def _process_timetable_chunk(chunk):
    """Verarbeitet einen Chunk von Stundenplan-Daten"""
    timetable_rows = []
    
    for entry in chunk:
        tag = entry.get('tag', '')
        stunde = entry.get('stunde', 0)
        
        faecher = entry.get('faecher', [])
        if not faecher:  # Leere Stunde
            timetable_rows.append({
                'Tag': tag,
                'Stunde': stunde,
                'Fach': '',
                'Lehrer': '',
                'Raum': ''
            })
        else:
            for fach_entry in faecher:
                # Problem mit 11re_e beheben (Fach normalisieren)
                fach = fach_entry.get('fach', '')
                if fach == '11re_e':
                    fach = '11re'
                
                timetable_rows.append({
                    'Tag': tag,
                    'Stunde': stunde,
                    'Fach': fach,
                    'Lehrer': fach_entry.get('lehrer', ''),
                    'Raum': fach_entry.get('raum', '')
                })
    
    return timetable_rows

@st.cache_data(ttl=CACHE_TTL)
def convert_courses_to_df(courses_data):
    """Kursdaten in DataFrame konvertieren mit Optimierungen und Fehlerbehandlung"""
    if not courses_data:
        return pd.DataFrame()
    
    # Performance-Messung starten
    start_time = time.time()
    
    try:
        # Für große Datensätze Parallelisierung verwenden
        if len(courses_data) > 500:
            # Anzahl der CPU-Kerne für die Parallelisierung
            num_cores = min(os.cpu_count() or 4, 8)  # Maximal 8 Kerne verwenden
            
            # Daten in Chunks aufteilen
            chunk_size = len(courses_data) // num_cores
            chunks = [courses_data[i:i+chunk_size] for i in range(0, len(courses_data), chunk_size)]
            
            # Parallele Verarbeitung mit ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                results = list(executor.map(_process_courses_chunk, chunks))
            
            # Ergebnisse zusammenführen
            course_rows = []
            for result in results:
                course_rows.extend(result)
        else:
            # Für kleinere Datensätze sequentielle Verarbeitung
            course_rows = _process_courses_chunk(courses_data)
        
        # DataFrame erstellen
        df = pd.DataFrame(course_rows)
        
        # Datentypen optimieren
        if not df.empty:
            if 'Kurs_ID' in df.columns:
                df['Kurs_ID'] = pd.to_numeric(df['Kurs_ID'], errors='coerce').fillna(0).astype(int)
            
            if 'Teilnehmerzahl' in df.columns:
                df['Teilnehmerzahl'] = pd.to_numeric(df['Teilnehmerzahl'], errors='coerce').fillna(0).astype(int)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Kurs-Konvertierung: {end_time - start_time:.2f} Sekunden")
        
        return df
    
    except Exception as e:
        logger.error(f"Fehler bei der Kurs-Konvertierung: {e}")
        # Fallback zur einfachen Verarbeitung ohne Parallelisierung
        course_rows = []
        
        for course in courses_data:
            try:
                kurs_id = course.get('kurs_id', '')
                kurs = course.get('kurs', '')
                kurs_type = course.get('kurs_type', '')
                participants_count = course.get('participants_count', 0)
                participants = course.get('participants', [])
                
                # Problem mit 11re_e beheben (Kurs normalisieren)
                if kurs == '11re_e':
                    kurs = '11re'
                
                course_rows.append({
                    'Kurs_ID': kurs_id,
                    'Kurs': kurs,
                    'Kurstyp': kurs_type,
                    'Teilnehmerzahl': participants_count,
                    'Teilnehmer': participants
                })
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung eines Kurses: {e}")
                continue
        
        return pd.DataFrame(course_rows)

def _process_courses_chunk(chunk):
    """Verarbeitet einen Chunk von Kursdaten"""
    course_rows = []
    
    for course in chunk:
        try:
            kurs_id = course.get('kurs_id', '')
            kurs = course.get('kurs', '')
            kurs_type = course.get('kurs_type', '')
            participants_count = course.get('participants_count', 0)
            participants = course.get('participants', [])
            
            # Problem mit 11re_e beheben (Kurs normalisieren)
            if kurs == '11re_e':
                kurs = '11re'
            
            course_rows.append({
                'Kurs_ID': kurs_id,
                'Kurs': kurs,
                'Kurstyp': kurs_type,
                'Teilnehmerzahl': participants_count,
                'Teilnehmer': participants
            })
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung eines Kurses: {e}")
            continue
    
    return course_rows

@st.cache_data(ttl=CACHE_TTL)
def convert_course_details_to_df(courses_data):
    """Konvertiert Kursdetails aus courses.json in DataFrame mit Fehlerbehandlung"""
    if not courses_data:
        return pd.DataFrame()
    
    # Performance-Messung starten
    start_time = time.time()
    
    try:
        course_rows = []
        
        for course in courses_data:
            try:
                kurs_id = course.get('kurs_id', '')
                kurs = course.get('kurs', '')
                lehrer_id = course.get('lehrer_id', '')
                lehrer = course.get('lehrer', {})
                lehrer_name = lehrer.get('name', '') if isinstance(lehrer, dict) else ''
                lehrer_kuerzel = lehrer.get('kuerzel', '') if isinstance(lehrer, dict) else ''
                
                # Problem mit 11re_e beheben (Kurs normalisieren)
                if kurs == '11re_e':
                    kurs = '11re'
                
                course_rows.append({
                    'Kurs_ID': kurs_id,
                    'Kurs': kurs,
                    'Lehrer_ID': lehrer_id,
                    'Lehrer_Name': lehrer_name,
                    'Lehrer_Kuerzel': lehrer_kuerzel
                })
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung eines Kurses: {e}")
                continue
        
        df = pd.DataFrame(course_rows)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Kursdetails-Konvertierung: {end_time - start_time:.2f} Sekunden")
        
        return df
    
    except Exception as e:
        logger.error(f"Fehler bei der Kursdetails-Konvertierung: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def convert_students_to_df(students_data):
    """Konvertiert Schülerdaten in DataFrame mit erweiterten Features und Fehlerbehandlung"""
    if not students_data:
        return pd.DataFrame()
    
    # Performance-Messung starten
    start_time = time.time()
    
    try:
        # Für große Datensätze Parallelisierung verwenden
        if len(students_data) > 500:
            # Anzahl der CPU-Kerne für die Parallelisierung
            num_cores = min(os.cpu_count() or 4, 8)  # Maximal 8 Kerne verwenden
            
            # Daten in Chunks aufteilen
            chunk_size = len(students_data) // num_cores
            chunks = [students_data[i:i+chunk_size] for i in range(0, len(students_data), chunk_size)]
            
            # Parallele Verarbeitung mit ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                results = list(executor.map(_process_students_chunk, chunks))
            
            # Ergebnisse zusammenführen
            student_rows = []
            for result in results:
                student_rows.extend(result)
        else:
            # Für kleinere Datensätze sequentielle Verarbeitung
            student_rows = _process_students_chunk(students_data)
        
        # DataFrame erstellen
        df = pd.DataFrame(student_rows)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Schüler-Konvertierung: {end_time - start_time:.2f} Sekunden")
        
        return df
    
    except Exception as e:
        logger.error(f"Fehler bei der Schüler-Konvertierung: {e}")
        # Fallback zur einfachen Verarbeitung
        student_rows = []
        
        for i, student in enumerate(students_data):
            try:
                # Schüler-ID und Name generieren, falls nicht vorhanden
                student_id = student.get('student_id', i)
                name = student.get('name', f"Schüler {i+1}")
                courses_str = student.get('courses', '')
                
                # Kursliste extrahieren und 11re_e korrigieren
                courses_list = [c.strip() for c in courses_str.split(',')]
                courses_list = ['11re' if c == '11re_e' else c for c in courses_list]
                
                student_rows.append({
                    'Schüler_ID': student_id,
                    'Name': name,
                    'Kurse_String': courses_str,
                    'Kurse_Liste': courses_list,
                    'Anzahl_Kurse': len(courses_list)
                })
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung eines Schülers: {e}")
                continue
        
        return pd.DataFrame(student_rows)

def _process_students_chunk(chunk):
    """Verarbeitet einen Chunk von Schülerdaten"""
    student_rows = []
    
    for i, student in enumerate(chunk):
        try:
            # Schüler-ID und Name generieren, falls nicht vorhanden
            student_id = student.get('student_id', i)
            name = student.get('name', f"Schüler {i+1}")
            courses_str = student.get('courses', '')
            
            # Kursliste extrahieren und 11re_e korrigieren
            courses_list = [c.strip() for c in courses_str.split(',')]
            courses_list = ['11re' if c == '11re_e' else c for c in courses_list]
            
            student_rows.append({
                'Schüler_ID': student_id,
                'Name': name,
                'Kurse_String': courses_str,
                'Kurse_Liste': courses_list,
                'Anzahl_Kurse': len(courses_list)
            })
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung eines Schülers: {e}")
            continue
    
    return student_rows

def parse_student_courses(courses_str):
    """Konvertiert einen Kurs-String in eine Liste von Kursen mit Normalisierung"""
    if not courses_str:
        return []
    
    courses = [c.strip() for c in courses_str.split(',')]
    
    # Problem mit 11re_e beheben
    courses = ['11re' if c == '11re_e' else c for c in courses]
    
    return courses

@st.cache_data(ttl=3600)
def generate_participant_course_matrix(students_df):
    """Erstellt eine Matrix, die zeigt, welche Schüler in welchen Kursen sind"""
    if students_df.empty:
        return pd.DataFrame()
    
    # Alle Kurse aus den Schülerdaten extrahieren
    all_courses = set()
    for courses_list in students_df['Kurse_Liste']:
        all_courses.update(courses_list)
    
    # Matrix erstellen: Zeilen = Schüler, Spalten = Kurse
    matrix_data = {course: [] for course in sorted(all_courses)}
    
    for _, student in students_df.iterrows():
        for course in all_courses:
            matrix_data[course].append(1 if course in student['Kurse_Liste'] else 0)
    
    # DataFrame erstellen
    df = pd.DataFrame(matrix_data, index=students_df['Name'])
    
    return df

@st.cache_data(ttl=3600)
def calculate_course_overlap(participant_course_matrix):
    """Berechnet die Überschneidungen zwischen Kursen"""
    if participant_course_matrix.empty:
        return pd.DataFrame()
    
    courses = participant_course_matrix.columns
    overlap_matrix = pd.DataFrame(index=courses, columns=courses)
    
    for course1 in courses:
        for course2 in courses:
            students_in_course1 = set(participant_course_matrix.index[participant_course_matrix[course1] == 1])
            students_in_course2 = set(participant_course_matrix.index[participant_course_matrix[course2] == 1])
            overlap = len(students_in_course1.intersection(students_in_course2))
            overlap_matrix.loc[course1, course2] = overlap
    
    return overlap_matrix

@st.cache_data(ttl=3600)
def merge_course_data(courses_df, course_details_df):
    """Verknüpft Kurs-Teilnehmerdaten mit Kursdetails"""
    if courses_df.empty or course_details_df.empty:
        return courses_df
    
    # Kursdetails mit Teilnehmerdaten verknüpfen
    merged_df = pd.merge(
        courses_df,
        course_details_df,
        left_on='Kurs',
        right_on='Kurs',
        how='left'
    )
    
    return merged_df

# Erweiterte Visualisierungsfunktionen
@st.cache_data(ttl=CACHE_TTL)
def generate_participant_course_matrix(students_df):
    """Erstellt eine Matrix, die zeigt, welche Schüler in welchen Kursen sind"""
    if students_df.empty:
        return pd.DataFrame()
    
    # Performance-Messung starten
    start_time = time.time()
    
    try:
        # Alle Kurse aus den Schülerdaten extrahieren
        all_courses = set()
        for courses_list in students_df['Kurse_Liste']:
            all_courses.update(courses_list)
        
        # Matrix erstellen: Zeilen = Schüler, Spalten = Kurse
        matrix_data = {course: [] for course in sorted(all_courses)}
        
        for _, student in students_df.iterrows():
            for course in all_courses:
                matrix_data[course].append(1 if course in student['Kurse_Liste'] else 0)
        
        # DataFrame erstellen
        df = pd.DataFrame(matrix_data, index=students_df['Name'])
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Matrix-Generierung: {end_time - start_time:.2f} Sekunden")
        
        return df
    
    except Exception as e:
        logger.error(f"Fehler bei der Matrix-Generierung: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def calculate_course_overlap(participant_course_matrix):
    """Berechnet die Überschneidungen zwischen Kursen mit Optimierungen"""
    if participant_course_matrix.empty:
        return pd.DataFrame()
    
    # Performance-Messung starten
    start_time = time.time()
    
    try:
        courses = participant_course_matrix.columns
        overlap_matrix = pd.DataFrame(index=courses, columns=courses)
        
        # Für jeden Kurs die Studenten finden, die ihn belegen
        course_students = {}
        for course in courses:
            course_students[course] = set(participant_course_matrix.index[participant_course_matrix[course] == 1])
        
        # Überschneidungen berechnen (optimierte Version)
        for i, course1 in enumerate(courses):
            students_in_course1 = course_students[course1]
            
            # Diagonale direkt setzen
            overlap_matrix.loc[course1, course1] = len(students_in_course1)
            
            # Nur die obere Dreiecksmatrix berechnen, dann spiegeln
            for j in range(i+1, len(courses)):
                course2 = courses[j]
                students_in_course2 = course_students[course2]
                
                overlap = len(students_in_course1.intersection(students_in_course2))
                overlap_matrix.loc[course1, course2] = overlap
                overlap_matrix.loc[course2, course1] = overlap  # Symmetrische Matrix
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Überschneidungsberechnung: {end_time - start_time:.2f} Sekunden")
        
        return overlap_matrix
    
    except Exception as e:
        logger.error(f"Fehler bei der Überschneidungsberechnung: {e}")
        return pd.DataFrame()

def plot_course_overlap_heatmap(overlap_matrix):
    """Erstellt eine interaktive Heatmap der Kursüberschneidungen"""
    if overlap_matrix.empty:
        return None
    
    try:
        fig = px.imshow(
            overlap_matrix,
            labels=dict(x="Kurs", y="Kurs", color="Anzahl überschneidender Schüler"),
            x=overlap_matrix.columns,
            y=overlap_matrix.index,
            color_continuous_scale='YlGnBu',
            title='Überschneidungen zwischen Kursen (Anzahl der gemeinsamen Schüler)'
        )
        
        fig.update_layout(
            height=800,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis={'side': 'bottom', 'tickangle': -45},
            yaxis={'autorange': 'reversed'},
            coloraxis_colorbar=dict(
                title="Anzahl Schüler",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
            ),
            font=dict(family="Open Sans, sans-serif"),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Open Sans, sans-serif"
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Fehler bei der Heatmap-Erstellung: {e}")
        return None

def create_network_graph(participant_course_matrix, threshold=3, focus_entity=None):
    """Erstellt ein interaktives Netzwerkdiagramm für Schüler-Kurs-Beziehungen"""
    if participant_course_matrix.empty:
        return None
    
    try:
        # Netzwerk erstellen
        G = nx.Graph()
        
        # Kurse als Knoten hinzufügen (mit Blau)
        for course in participant_course_matrix.columns:
            G.add_node(course, bipartite=0, type='course', size=10)
        
        # Schüler als Knoten hinzufügen (mit Rot)
        for student in participant_course_matrix.index:
            G.add_node(student, bipartite=1, type='student', size=5)
        
        # Kanten zwischen Schülern und Kursen hinzufügen
        for student in participant_course_matrix.index:
            for course in participant_course_matrix.columns:
                if participant_course_matrix.loc[student, course] == 1:
                    G.add_edge(student, course, weight=1)
        
        # Wenn ein Fokus-Element angegeben ist, nur verwandte Knoten anzeigen
        if focus_entity:
            if focus_entity in G:
                # Nachbarn des Fokus-Elements
                neighbors = list(G.neighbors(focus_entity))
                # Für jeden Nachbarn auch dessen Nachbarn hinzufügen (für Kurs-Schüler-Kurs-Verbindungen)
                second_neighbors = []
                for neighbor in neighbors:
                    second_neighbors.extend(list(G.neighbors(neighbor)))
                # Alle relevanten Knoten
                nodes_to_keep = [focus_entity] + neighbors + second_neighbors
                G = G.subgraph(nodes_to_keep)
        
        # Layout berechnen (Fruchterman-Reingold für bessere Verteilung)
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Knoten und Kanten für Plotly vorbereiten
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            if G.nodes[node]['type'] == 'course':
                node_color.append(COLOR_PALETTE['primary'])  # Blau für Kurse
                node_size.append(15)
            else:
                node_color.append(COLOR_PALETTE['danger'])  # Rot für Schüler
                node_size.append(10)
        
        # Kanten vorbereiten
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Kanten zeichnen
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#94A3B8'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Knoten zeichnen
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=1, color='#CBD5E1')
            ),
            text=node_text,
            hoverinfo='text'
        )
        
        # Layout erstellen
        layout = go.Layout(
            title='Schüler-Kurs-Netzwerk',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Open Sans, sans-serif"),
            annotations=[
                dict(
                    text="Blau: Kurse | Rot: Schüler",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.01, y=-0.01,
                    font=dict(family="Open Sans, sans-serif")
                )
            ]
        )
        
        # Figure erstellen
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        
        return fig
    
    except Exception as e:
        logger.error(f"Fehler bei der Netzwerkdiagramm-Erstellung: {e}")
        return None

def create_3d_course_visualization(overlap_matrix):
    """Erstellt eine 3D-Visualisierung der Kursüberschneidungen"""
    if overlap_matrix.empty:
        return None
    
    try:
        # Diagonale (Selbstüberschneidung) auf 0 setzen
        np.fill_diagonal(overlap_matrix.values, 0)
        
        # 3D-Scatter-Plot vorbereiten
        fig = go.Figure()
        
        # Multi-dimensional scaling (MDS) für 3D-Koordinaten
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        
        # Distanzmatrix erstellen (inverse der Überlappung)
        max_overlap = overlap_matrix.max().max()
        if max_overlap > 0:
            distance_matrix = 1 - (overlap_matrix / max_overlap)
            positions = mds.fit_transform(distance_matrix)
        else:
            # Wenn keine Überlappungen, zufällige Positionen verwenden
            positions = np.random.rand(len(overlap_matrix), 3)
        
        # Cluster basierend auf Überlappungen
        clustering = AgglomerativeClustering(
            n_clusters=min(5, len(overlap_matrix)),
            affinity='precomputed',
            linkage='average'
        ).fit(distance_matrix)
        
        # Farben und Größen basierend auf Clustern und Überlappungssummen
        colors = [
            COLOR_PALETTE['primary'],
            COLOR_PALETTE['secondary'],
            COLOR_PALETTE['danger'],
            COLOR_PALETTE['success'],
            COLOR_PALETTE['warning']
        ]
        
        sizes = overlap_matrix.sum(axis=1).values  # Gesamte Überlappung pro Kurs
        max_size = max(sizes) if len(sizes) > 0 else 1
        normalized_sizes = [20 + (s / max_size) * 30 for s in sizes]  # Größen zwischen 20 und 50
        
        # 3D-Scatter-Plot erstellen
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            marker=dict(
                size=normalized_sizes,
                color=[colors[c % len(colors)] for c in clustering.labels_],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=overlap_matrix.index,
            hoverinfo='text',
            hovertext=[f"Kurs: {course}<br>Gesamtüberlappung: {int(sizes[i])}" 
                      for i, course in enumerate(overlap_matrix.index)]
        ))
        
        # Layout anpassen
        fig.update_layout(
            title='3D-Visualisierung der Kursüberschneidungen',
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            font=dict(family="Open Sans, sans-serif")
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Fehler bei der 3D-Visualisierung: {e}")
        return None

def generate_student_timetable(student_name, students_df, timetable_df, course_participants_df=None):
    """Generiert einen individuellen Stundenplan für einen Schüler"""
    if students_df.empty or timetable_df.empty:
        return pd.DataFrame()
    
    try:
        # Kurse des Schülers finden
        student_data = students_df[students_df['Name'] == student_name]
        if student_data.empty:
            return pd.DataFrame()
        
        student_courses = student_data.iloc[0]['Kurse_Liste']
        
        # Lowercase und Uppercase Varianten der Kurse berücksichtigen
        student_courses_lower = [course.lower() for course in student_courses]
        
        # Stundenplan filtern auf die Kurse des Schülers
        student_timetable = timetable_df[
            timetable_df['Fach'].apply(lambda x: x.lower() if isinstance(x, str) else '').isin(student_courses_lower) |
            timetable_df['Fach'].isin(student_courses)
        ]
        
        # Leeres DataFrame für alle Tage und Stunden erstellen
        weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag']
        hours = range(0, 10)  # 0 bis 9. Stunde
        
        full_timetable = pd.DataFrame(
            [(day, hour) for day in weekdays for hour in hours],
            columns=['Tag', 'Stunde']
        )
        
        # Mit dem Stundenplan des Schülers zusammenführen
        merged_timetable = pd.merge(
            full_timetable,
            student_timetable,
            on=['Tag', 'Stunde'],
            how='left'
        )
        
        # Nach Tag und Stunde sortieren
        weekday_order = {
            'Montag': 0, 
            'Dienstag': 1, 
            'Mittwoch': 2, 
            'Donnerstag': 3, 
            'Freitag': 4
        }
        
        merged_timetable['Tag_Order'] = merged_timetable['Tag'].map(weekday_order)
        merged_timetable = merged_timetable.sort_values(['Tag_Order', 'Stunde']).drop('Tag_Order', axis=1)
        
        return merged_timetable
    
    except Exception as e:
        logger.error(f"Fehler bei der Stundenplan-Generierung: {e}")
        return pd.DataFrame()

def visualize_student_timetable(student_timetable):
    """Erstellt eine ansprechende Visualisierung des Stundenplans eines Schülers"""
    if student_timetable.empty:
        return None
    
    try:
        # Pivot-Tabelle für den Stundenplan erstellen
        pivot = student_timetable.pivot_table(
            index='Stunde', 
            columns='Tag', 
            values='Fach', 
            aggfunc='first'
        ).fillna('')
        
        # Wochentage in richtige Reihenfolge bringen
        wochentage = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag']
        pivot = pivot.reindex(columns=wochentage)
        
        # Farbkodierung für Fächer erstellen
        unique_subjects = [subj for subj in student_timetable['Fach'].unique() if subj]
        subject_colors = {}
        
        # Farbpalette für Fächer
        color_palette = [
            COLOR_PALETTE['primary'],
            COLOR_PALETTE['secondary'],
            COLOR_PALETTE['tertiary'],
            COLOR_PALETTE['warning'],
            COLOR_PALETTE['danger'],
            COLOR_PALETTE['info'],
            COLOR_PALETTE['success'],
        ]
        
        for i, subject in enumerate(unique_subjects):
            subject_colors[subject] = color_palette[i % len(color_palette)]
        
        # Annotations für die Zellen erstellen
        annotations = []
        
        for i, stunde in enumerate(pivot.index):
            for j, tag in enumerate(pivot.columns):
                fach = pivot.loc[stunde, tag]
                if fach:
                    # Informationen zum Fach aus dem Originaldatensatz abrufen
                    fach_info = student_timetable[
                        (student_timetable['Tag'] == tag) & 
                        (student_timetable['Stunde'] == stunde) & 
                        (student_timetable['Fach'] == fach)
                    ]
                    
                    if not fach_info.empty:
                        lehrer = fach_info.iloc[0]['Lehrer']
                        raum = fach_info.iloc[0]['Raum']
                        
                        text = f"{fach}<br>{lehrer}<br>{raum}"
                    else:
                        text = fach
                    
                    annotations.append(dict(
                        x=j,
                        y=i,
                        text=text,
                        showarrow=False,
                        font=dict(
                            color='white' if fach else 'black',
                            size=10
                        )
                    ))
        
        # Heatmap erstellen mit Farbkodierung nach Fach
        z = []
        hover_texts = []
        
        for i, stunde in enumerate(pivot.index):
            z_row = []
            hover_row = []
            
            for j, tag in enumerate(pivot.columns):
                fach = pivot.loc[stunde, tag]
                
                # Zahlenwert für die Heatmap (für Farbkodierung)
                if fach:
                    z_row.append(list(unique_subjects).index(fach) + 1)
                    
                    # Informationen für den Hover-Text
                    fach_info = student_timetable[
                        (student_timetable['Tag'] == tag) & 
                        (student_timetable['Stunde'] == stunde) & 
                        (student_timetable['Fach'] == fach)
                    ]
                    
                    if not fach_info.empty:
                        lehrer = fach_info.iloc[0]['Lehrer']
                        raum = fach_info.iloc[0]['Raum']
                        hover_row.append(f"Fach: {fach}<br>Lehrer: {lehrer}<br>Raum: {raum}")
                    else:
                        hover_row.append(f"Fach: {fach}")
                else:
                    z_row.append(0)
                    hover_row.append("Keine Stunde")
            
            z.append(z_row)
            hover_texts.append(hover_row)
        
        # Colorscale basierend auf den Fächern erstellen
        colorscale = [[0, COLOR_PALETTE['light']]]  # Hellgrau für leere Zellen
        
        for i, subject in enumerate(unique_subjects):
            colorscale.append([(i + 1) / (len(unique_subjects) + 1), subject_colors[subject]])
            colorscale.append([(i + 1.9) / (len(unique_subjects) + 1), subject_colors[subject]])
        
        # Heatmap erstellen
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=pivot.columns,
            y=pivot.index,
            colorscale=colorscale,
            showscale=False,
            text=hover_texts,
            hoverinfo='text'
        ))
        
        # Layout anpassen
        fig.update_layout(
            title="Individueller Stundenplan",
            xaxis=dict(side='top', title=''),
            yaxis=dict(title='Stunde', dtick=1),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40),
            font=dict(family="Open Sans, sans-serif"),
            annotations=annotations
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Fehler bei der Stundenplan-Visualisierung: {e}")
        return None

# Datenbankanbindungsfunktionen
def connect_to_mongodb(connection_string):
    """Stellt eine Verbindung zu MongoDB her"""
    try:
        client = pymongo.MongoClient(connection_string)
        return client, True
    except Exception as e:
        st.error(f"Fehler bei der Verbindung zu MongoDB: {e}")
        return None, False

def connect_to_sqlite(db_path):
    """Stellt eine Verbindung zu SQLite her"""
    try:
        conn = sqlite3.connect(db_path)
        return conn, True
    except Exception as e:
        st.error(f"Fehler bei der Verbindung zu SQLite: {e}")
        return None, False

def save_to_mongodb(data, collection_name, client):
    """Speichert Daten in MongoDB"""
    try:
        db = client['kursplan_analyse']
        collection = db[collection_name]
        
        # Wenn data ein DataFrame ist, in Dict konvertieren
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        
        if isinstance(data, list):
            result = collection.insert_many(data)
            return len(result.inserted_ids), True
        else:
            result = collection.insert_one(data)
            return 1, True
    except Exception as e:
        st.error(f"Fehler beim Speichern in MongoDB: {e}")
        return 0, False

def save_to_sqlite(df, table_name, conn):
    """Speichert DataFrame in SQLite"""
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        return len(df), True
    except Exception as e:
        st.error(f"Fehler beim Speichern in SQLite: {e}")
        return 0, False

# Profilsystem für verschiedene Benutzerrollen
class UserProfileManager:
    """Verwaltet Benutzerprofile und Berechtigungen"""
    
    def __init__(self, profiles_file="user_profiles.json"):
        self.profiles_file = profiles_file
        self.profiles = self._load_profiles()
        self.current_user = None
        self.is_authenticated = False
    
    def _load_profiles(self):
        """Lädt Benutzerprofile aus der Datei"""
        try:
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # Standard-Benutzerprofile erstellen
                default_profiles = {
                    "admin": {
                        "name": "Administrator",
                        "password": self._hash_password("admin"),  # In Produktion stärkeres Passwort verwenden
                        "role": "admin",
                        "permissions": ["view", "edit", "delete", "admin"],
                        "settings": {}
                    },
                    "lehrer": {
                        "name": "Lehrer",
                        "password": self._hash_password("lehrer"),
                        "role": "lehrer",
                        "permissions": ["view", "edit"],
                        "settings": {}
                    },
                    "verwaltung": {
                        "name": "Verwaltung",
                        "password": self._hash_password("verwaltung"),
                        "role": "verwaltung",
                        "permissions": ["view", "edit", "admin"],
                        "settings": {}
                    },
                    "gast": {
                        "name": "Gast",
                        "password": self._hash_password("gast"),
                        "role": "gast",
                        "permissions": ["view"],
                        "settings": {}
                    }
                }
                
                # Speichere Standardprofile
                with open(self.profiles_file, "w", encoding="utf-8") as f:
                    json.dump(default_profiles, f, ensure_ascii=False, indent=4)
                
                return default_profiles
        except Exception as e:
            logger.error(f"Fehler beim Laden der Benutzerprofile: {e}")
            # Notfall-Rückgabe eines Admin-Profils
            return {
                "admin": {
                    "name": "Administrator",
                    "password": self._hash_password("admin"),
                    "role": "admin",
                    "permissions": ["view", "edit", "delete", "admin"],
                    "settings": {}
                }
            }
    
    def _save_profiles(self):
        """Speichert Benutzerprofile in der Datei"""
        try:
            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Benutzerprofile: {e}")
            return False
    
    def _hash_password(self, password):
        """Erstellt einen Hash für ein Passwort"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def login(self, username, password):
        """Authentifiziert einen Benutzer"""
        if username in self.profiles:
            stored_hash = self.profiles[username]["password"]
            if self._hash_password(password) == stored_hash:
                self.current_user = username
                self.is_authenticated = True
                return True
        
        return False
    
    def logout(self):
        """Meldet den aktuellen Benutzer ab"""
        self.current_user = None
        self.is_authenticated = False
    
    def get_current_user(self):
        """Gibt den aktuellen Benutzer zurück"""
        if self.is_authenticated and self.current_user in self.profiles:
            return self.profiles[self.current_user]
        return None
    
    def has_permission(self, permission):
        """Prüft, ob der aktuelle Benutzer eine bestimmte Berechtigung hat"""
        if not self.is_authenticated:
            return False
        
        if self.current_user in self.profiles:
            return permission in self.profiles[self.current_user]["permissions"]
        
        return False
    
    def add_user(self, username, name, password, role, permissions=None):
        """Fügt einen neuen Benutzer hinzu"""
        if not self.is_authenticated or not self.has_permission("admin"):
            return False
        
        if username in self.profiles:
            return False
        
        if permissions is None:
            permissions = ["view"]
        
        self.profiles[username] = {
            "name": name,
            "password": self._hash_password(password),
            "role": role,
            "permissions": permissions,
            "settings": {}
        }
        
        return self._save_profiles()
    
    def update_user(self, username, data):
        """Aktualisiert die Daten eines Benutzers"""
        if not self.is_authenticated or not self.has_permission("admin"):
            if self.current_user != username:
                return False
        
        if username not in self.profiles:
            return False
        
        # Passwort separat behandeln
        if "password" in data:
            data["password"] = self._hash_password(data["password"])
        
        # Daten aktualisieren
        for key, value in data.items():
            self.profiles[username][key] = value
        
        return self._save_profiles()
    
    def delete_user(self, username):
        """Löscht einen Benutzer"""
        if not self.is_authenticated or not self.has_permission("admin"):
            return False
        
        if username not in self.profiles:
            return False
        
        del self.profiles[username]
        return self._save_profiles()
    
    def get_user_settings(self, username=None):
        """Gibt die Einstellungen eines Benutzers zurück"""
        if username is None:
            username = self.current_user
        
        if not self.is_authenticated or username not in self.profiles:
            return {}
        
        return self.profiles[username].get("settings", {})
    
    def save_user_settings(self, settings, username=None):
        """Speichert die Einstellungen eines Benutzers"""
        if username is None:
            username = self.current_user
        
        if not self.is_authenticated or username not in self.profiles:
            return False
        
        self.profiles[username]["settings"] = settings
        return self._save_profiles()

# PWA-Funktionalität (Progressive Web App)
def setup_pwa():
    """Richtet die PWA-Funktionalität ein"""
    
    # Manifest-Datei erstellen
    manifest_path = os.path.join("static", "manifest.json")
    
    # Erstelle static-Verzeichnis, falls es nicht existiert
    if not os.path.exists("static"):
        os.makedirs("static")
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(PWA_MANIFEST, f, ensure_ascii=False, indent=4)
    
    # Service Worker-Datei erstellen
    sw_path = os.path.join("static", "service-worker.js")
    
    sw_content = """
    // Service Worker für die Kursplan-Analyse-App
    const CACHE_NAME = 'kursplan-analyse-cache-v1';
    const urlsToCache = [
        '/',
        '/static/manifest.json',
        '/static/icon-192x192.png',
        '/static/icon-512x512.png'
    ];

    self.addEventListener('install', function(event) {
        event.waitUntil(
            caches.open(CACHE_NAME)
                .then(function(cache) {
                    console.log('Cache geöffnet');
                    return cache.addAll(urlsToCache);
                })
        );
    });

    self.addEventListener('fetch', function(event) {
        event.respondWith(
            caches.match(event.request)
                .then(function(response) {
                    // Cache-Hit - zurückgeben
                    if (response) {
                        return response;
                    }

                    // Kein Cache-Hit - Anfrage klonen
                    const fetchRequest = event.request.clone();

                    return fetch(fetchRequest).then(
                        function(response) {
                            // Prüfen, ob wir eine gültige Antwort erhalten haben
                            if(!response || response.status !== 200 || response.type !== 'basic') {
                                return response;
                            }

                            // Antwort klonen
                            const responseToCache = response.clone();

                            caches.open(CACHE_NAME)
                                .then(function(cache) {
                                    cache.put(event.request, responseToCache);
                                });

                            return response;
                        }
                    );
                })
        );
    });

    self.addEventListener('activate', function(event) {
        const cacheWhitelist = [CACHE_NAME];

        event.waitUntil(
            caches.keys().then(function(cacheNames) {
                return Promise.all(
                    cacheNames.map(function(cacheName) {
                        if (cacheWhitelist.indexOf(cacheName) === -1) {
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
        );
    });
    """
    
    with open(sw_path, "w", encoding="utf-8") as f:
        f.write(sw_content)
    
    # PWA-Metadaten in den HTML-Header einfügen
    pwa_meta = """
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#2563EB">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="Kursplan-Analyse">
    <link rel="apple-touch-icon" href="/static/icon-192x192.png">
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                navigator.serviceWorker.register('/static/service-worker.js').then(
                    function(registration) {
                        console.log('ServiceWorker registration successful with scope: ', registration.scope);
                    },
                    function(err) {
                        console.log('ServiceWorker registration failed: ', err);
                    }
                );
            });
        }
    </script>
    """
    
    st.markdown(pwa_meta, unsafe_allow_html=True)

# Responsive Design-Funktionen
def apply_responsive_styles():
    """Wendet responsive Styles für verschiedene Geräte an"""
    
    responsive_css = """
    <style>
    /* Basis-Styles */
    :root {
        --primary-color: #2563EB;
        --secondary-color: #7C3AED;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --info-color: #3B82F6;
        --light-color: #F3F4F6;
        --dark-color: #1F2937;
        --white: #FFFFFF;
        --black: #000000;
        --background-color: #F9FAFB;
        --text-color: #111827;
        --muted-color: #6B7280;
        --border-radius: 0.5rem;
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Mobile Styles (bis 576px) */
    @media (max-width: 576px) {
        .main-header {
            font-size: 1.5rem !important;
        }
        
        .sub-header {
            font-size: 1.2rem !important;
        }
        
        .section-header {
            font-size: 1.1rem !important;
        }
        
        .card {
            padding: 0.8rem !important;
        }
        
        .metric-value {
            font-size: 1.5rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.7rem !important;
            padding: 0.3rem 0.5rem !important;
        }
        
        .plot-container {
            height: 250px !important;
        }
        
        .dataframe {
            font-size: 0.7rem !important;
        }
    }
    
    /* Tablet Styles (576px - 992px) */
    @media (min-width: 577px) and (max-width: 992px) {
        .main-header {
            font-size: 1.8rem !important;
        }
        
        .sub-header {
            font-size: 1.4rem !important;
        }
        
        .card {
            padding: 1rem !important;
        }
        
        .metric-value {
            font-size: 1.6rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem !important;
        }
        
        .plot-container {
            height: 400px !important;
        }
    }
    
    /* Desktop Styles (ab 992px) */
    @media (min-width: 993px) {
        .main-header {
            font-size: 2rem !important;
        }
        
        .sub-header {
            font-size: 1.6rem !important;
        }
        
        .card {
            padding: 1.2rem !important;
        }
        
        .metric-value {
            font-size: 1.8rem !important;
        }
        
        .plot-container {
            height: 500px !important;
        }
    }
    
    /* Styles für bessere Touch-Interaktion auf mobilen Geräten */
    @media (pointer: coarse) {
        .stButton > button {
            padding: 0.7rem 1rem !important;
            min-height: 44px !important;  /* Empfohlen für Touch-Targets */
        }
        
        .stSelectbox [data-baseweb=select] {
            min-height: 44px !important;
        }
        
        .stCheckbox > div {
            min-height: 44px !important;
        }
    }
    
    /* Druckoptimierungen */
    @media print {
        .stButton, .stSidebar, .stCheckbox, .stSelectbox, .stFileUploader {
            display: none !important;
        }
        
        .main .block-container {
            max-width: 100% !important;
            padding: 0 !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            break-after: avoid;
        }
        
        table, figure {
            break-inside: avoid;
        }
    }
    </style>
    """
    
    st.markdown(responsive_css, unsafe_allow_html=True)

# Erkennung der Gerätegröße
def detect_device():
    """Erkennt den Gerätetyp des Benutzers über JavaScript"""
    
    device_detection_js = """
    <script>
    // Geräteerkennung
    function detectDevice() {
        let width = window.innerWidth;
        let deviceType = "";
        
        if (width <= 576) {
            deviceType = "mobile";
        } else if (width <= 992) {
            deviceType = "tablet";
        } else {
            deviceType = "desktop";
        }
        
        // Speichere im localStorage für Streamlit
        localStorage.setItem('deviceType', deviceType);
        localStorage.setItem('screenWidth', width);
        
        // Füge CSS-Klasse zum Body hinzu
        document.body.classList.remove('mobile', 'tablet', 'desktop');
        document.body.classList.add(deviceType);
    }
    
    // Ausführen beim Laden und Größenänderung
    window.addEventListener('load', detectDevice);
    window.addEventListener('resize', detectDevice);
    </script>
    """
    
    st.markdown(device_detection_js, unsafe_allow_html=True)
    
    # Standardwert zurückgeben (tatsächlicher Wert wird durch JavaScript gesetzt)
    return "desktop"

# Hauptanwendung
def main():
    st.markdown('<div class="main-header">📊 Kursplan- und Stundenplan-Analyse</div>', unsafe_allow_html=True)
    
    # Seitenleiste für Dateiauswahl
    st.sidebar.title("Dateiauswahl")
    
    # Pfade festlegen
    default_timetable_path = "json/timetable.json"
    default_courses_path = "json/course_participants.json"
    default_students_path = "json/students.json"
    
    timetable_path = st.sidebar.text_input(
        "Pfad zur Stundenplan-Datei (JSON):",
        value=default_timetable_path,
        help="Geben Sie den relativen oder absoluten Pfad zur JSON-Datei mit den Stundenplan-Daten an."
    )
    
    courses_path = st.sidebar.text_input(
        "Pfad zur Kursdaten-Datei (JSON):",
        value=default_courses_path,
        help="Geben Sie den relativen oder absoluten Pfad zur JSON-Datei mit den Kursdaten an."
    )
    
    students_path = st.sidebar.text_input(
        "Pfad zur Schülerdaten-Datei (JSON):",
        value=default_students_path,
        help="Geben Sie den relativen oder absoluten Pfad zur JSON-Datei mit den Schülerdaten an."
    )
    
    # Daten laden
    with st.spinner("Daten werden geladen..."):
        timetable_data = load_json_file(timetable_path)
        courses_data = load_json_file(courses_path)
        # students_data = load_json_file(students_path)  # Für spätere Verwendung
    
    if not timetable_data or not courses_data:
        st.error("Eine oder mehrere Dateien konnten nicht geladen werden. Bitte überprüfen Sie die Pfade.")
        return
    
    # Daten in DataFrames konvertieren
    timetable_df = convert_timetable_to_df(timetable_data)
    courses_df = convert_courses_to_df(courses_data)
    
    # Tabs für verschiedene Analysen
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Übersicht", 
        "🗓️ Stundenplan-Analyse", 
        "👥 Kurs-Analyse", 
        "� Personensuche", 
        "🛠️ Developer"
    ])
    
    # Tab 1: Übersicht
    with tab1:
        st.markdown('<div class="sub-header">📈 Datenübersicht</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Stundenplan")
            st.write(f"**Anzahl der Einträge:** {len(timetable_df)}")
            st.write(f"**Anzahl der Tage:** {timetable_df['Tag'].nunique()}")
            st.write(f"**Anzahl der Stunden:** {timetable_df['Stunde'].nunique()}")
            st.write(f"**Anzahl der Fächer:** {timetable_df['Fach'].nunique()}")
            st.write(f"**Anzahl der Lehrer:** {timetable_df['Lehrer'].nunique()}")
            st.write(f"**Anzahl der Räume:** {timetable_df['Raum'].nunique()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Kurse")
            st.write(f"**Anzahl der Kurse:** {len(courses_df)}")
            st.write(f"**Anzahl der Grundkurse:** {len(courses_df[courses_df['Kurstyp'] == 'Grundkurs'])}")
            st.write(f"**Anzahl der Leistungskurse:** {len(courses_df[courses_df['Kurstyp'] == 'Leistungskurs'])}")
            
            unique_participants = set()
            for participants in courses_df['Teilnehmer']:
                unique_participants.update(participants)
            
            st.write(f"**Anzahl der einzigartigen Teilnehmer:** {len(unique_participants)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">📊 Schnellstatistiken</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 10 Kurse nach Teilnehmerzahl
            top_courses_fig = plot_course_participants_bar(courses_df, 10)
            st.plotly_chart(top_courses_fig, use_container_width=True)
        
        with col2:
            # Stundenplanauslastung
            timetable_heatmap = create_timetable_heatmap(timetable_df)
            st.plotly_chart(timetable_heatmap, use_container_width=True)
    
    # Tab 2: Stundenplan-Analyse
    with tab2:
        st.markdown('<div class="sub-header">🗓️ Stundenplandetails</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filter für den Stundenplan
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Stundenplan filtern")
            
            # Filter-Optionen
            tage = ['Alle'] + sorted(timetable_df['Tag'].unique().tolist())
            selected_tag = st.selectbox("Tag auswählen:", tage)
            
            stunden = ['Alle'] + sorted(timetable_df['Stunde'].unique().tolist())
            selected_stunde = st.selectbox("Stunde auswählen:", stunden)
            
            faecher = ['Alle'] + sorted([f for f in timetable_df['Fach'].unique() if f])
            selected_fach = st.selectbox("Fach auswählen:", faecher)
            
            lehrer = ['Alle'] + sorted([l for l in timetable_df['Lehrer'].unique() if l])
            selected_lehrer = st.selectbox("Lehrer auswählen:", lehrer)
            
            raeume = ['Alle'] + sorted([r for r in timetable_df['Raum'].unique() if r])
            selected_raum = st.selectbox("Raum auswählen:", raeume)
            
            # Filter anwenden
            filtered_df = timetable_df.copy()
            
            if selected_tag != 'Alle':
                filtered_df = filtered_df[filtered_df['Tag'] == selected_tag]
                
            if selected_stunde != 'Alle':
                filtered_df = filtered_df[filtered_df['Stunde'] == selected_stunde]
                
            if selected_fach != 'Alle':
                filtered_df = filtered_df[filtered_df['Fach'] == selected_fach]
                
            if selected_lehrer != 'Alle':
                filtered_df = filtered_df[filtered_df['Lehrer'] == selected_lehrer]
                
            if selected_raum != 'Alle':
                filtered_df = filtered_df[filtered_df['Raum'] == selected_raum]
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Gefilterter Stundenplan
            if not filtered_df.empty:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Gefilterter Stundenplan")
                st.dataframe(filtered_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Keine Daten für die ausgewählten Filter gefunden.")
        
        with col2:
            # Lehrerarbeitsbelastung
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Lehrerbelastung")
            
            teacher_workload = analyze_teacher_workload(timetable_df)
            
            top_n_teachers = st.slider("Anzahl der anzuzeigenden Lehrer:", 5, 20, 10)
            
            fig = px.bar(
                teacher_workload.head(top_n_teachers),
                x='Lehrer',
                y='Anzahl_Stunden',
                title=f'Top {top_n_teachers} Lehrer nach Unterrichtsstunden',
                labels={'Anzahl_Stunden': 'Anzahl der Stunden', 'Lehrer': 'Lehrerkürzel'}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Raumnutzung
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Raumnutzung")
            
            room_usage = get_room_usage(timetable_df)
            
            top_n_rooms = st.slider("Anzahl der anzuzeigenden Räume:", 5, 20, 10)
            
            fig = px.bar(
                room_usage.head(top_n_rooms),
                x='Raum',
                y='Nutzungshäufigkeit',
                title=f'Top {top_n_rooms} Räume nach Nutzungshäufigkeit',
                labels={'Nutzungshäufigkeit': 'Häufigkeit der Nutzung', 'Raum': 'Raumnummer'}
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Kurs-Analyse
    with tab3:
        st.markdown('<div class="sub-header">👥 Kursdetails</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Kursdetails
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Kursauswahl")
            
            course_options = courses_df['Kurs'].tolist()
            selected_course = st.selectbox("Kurs auswählen:", course_options)
            
            selected_course_data = courses_df[courses_df['Kurs'] == selected_course].iloc[0]
            
            st.write(f"**Kurs-ID:** {selected_course_data['Kurs_ID']}")
            st.write(f"**Kurstyp:** {selected_course_data['Kurstyp']}")
            st.write(f"**Teilnehmerzahl:** {selected_course_data['Teilnehmerzahl']}")
            
            st.subheader("Teilnehmerliste")
            if selected_course_data['Teilnehmer']:
                st.write(", ".join(selected_course_data['Teilnehmer']))
            else:
                st.info("Keine Teilnehmerdaten verfügbar für diesen Kurs.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Kurs im Stundenplan anzeigen
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Kurs im Stundenplan")
            
            # Lowercase und Uppercase Variante des Kurses berücksichtigen
            course_in_timetable = timetable_df[
                (timetable_df['Fach'].str.lower() == selected_course.lower()) | 
                (timetable_df['Fach'] == selected_course)
            ]
            
            if not course_in_timetable.empty:
                st.dataframe(course_in_timetable[['Tag', 'Stunde', 'Lehrer', 'Raum']], use_container_width=True)
            else:
                st.info(f"Der Kurs '{selected_course}' wurde im Stundenplan nicht gefunden.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Kursstatistiken
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Kursstatistiken")
            
            # Durchschnittliche Teilnehmerzahl nach Kurstyp
            avg_participants = courses_df.groupby('Kurstyp')['Teilnehmerzahl'].mean().reset_index()
            
            fig = px.bar(
                avg_participants,
                x='Kurstyp',
                y='Teilnehmerzahl',
                title='Durchschnittliche Teilnehmerzahl nach Kurstyp',
                labels={'Teilnehmerzahl': 'Durchschnittliche Anzahl', 'Kurstyp': 'Kurstyp'},
                color='Kurstyp',
                color_discrete_map={
                    'Leistungskurs': '#2563EB',
                    'Grundkurs': '#7C3AED'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Verteilung der Teilnehmerzahlen
            fig = px.histogram(
                courses_df,
                x='Teilnehmerzahl',
                color='Kurstyp',
                title='Verteilung der Kursteilnehmerzahlen',
                labels={'Teilnehmerzahl': 'Anzahl der Teilnehmer', 'count': 'Anzahl der Kurse'},
                color_discrete_map={
                    'Leistungskurs': '#2563EB',
                    'Grundkurs': '#7C3AED'
                },
                nbins=15
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: Personensuche und Analyse
    with tab4:
        st.markdown('<div class="sub-header">� Personensuche & Analyse</div>', unsafe_allow_html=True)
        
        # Studierende aus JSON-Datei laden
        try:
            with open('json/students.json', 'r', encoding='utf-8') as file:
                students_data = json.load(file)
            
            # DataFrame erstellen
            students_df = pd.DataFrame(students_data)
            
            # Spaltenname standardisieren (groß schreiben)
            students_df.rename(columns={'name': 'Name'}, inplace=True)
            
            # Kursliste als separate Spalte für einfacheren Zugriff
            # Das courses-Feld ist ein String mit kommagetrennten Kursen
            students_df['Kurse_Liste'] = students_df['courses'].apply(lambda courses: [course.strip() for course in str(courses).split(',')])
            
            if not students_df.empty:
                # Tabs für verschiedene Suchoptionen
                search_tabs = st.tabs(["🔍 Namenssuche", "📚 Kursbasierte Suche", "🔄 Personen-Heatmap", "🌐 Schülernetzwerk"])
                
                # Tab 1: Namenssuche
                with search_tabs[0]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Suche nach Schülernamen")
                    
                    # Suchfeld
                    search_query = st.text_input("Namen eingeben:", 
                                                placeholder="Vor- oder Nachname eingeben...")
                    
                    if search_query:
                        # Suche durchführen
                        matched_students = search_students_by_name(students_df, search_query)
                        
                        if not matched_students.empty:
                            st.success(f"{len(matched_students)} Schüler gefunden.")
                            
                            # Ergebnisse anzeigen
                            for _, student in matched_students.iterrows():
                                with st.expander(f"{student['Name']} - {len(student['Kurse_Liste'])} Kurse"):
                                    st.write(f"**ID:** {student['student_id']}")
                                    st.write("**Kurse:**")
                                    for course in student['Kurse_Liste']:
                                        st.write(f"- {course}")
                                    
                                    # Stundenplan des Schülers anzeigen
                                    st.write("**Stundenplan:**")
                                    student_timetable = generate_student_timetable(student['student_id'], 
                                                                                  students_df, 
                                                                                  courses_df)
                                    
                                    if not student_timetable.empty:
                                        fig = visualize_student_timetable(student_timetable)
                                        if fig:
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.info("Stundenplan konnte nicht visualisiert werden.")
                                    else:
                                        st.info("Kein Stundenplan verfügbar.")
                        else:
                            st.info("Keine Schüler mit diesem Namen gefunden.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Tab 2: Kursbasierte Suche
                with search_tabs[1]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Suche nach belegten Kursen")
                    
                    # Alle verfügbaren Kurse
                    all_courses = set()
                    for courses_list in students_df['Kurse_Liste']:
                        all_courses.update(courses_list)
                    
                    # Sortierte Liste für die Auswahl
                    sorted_courses = sorted(list(all_courses))
                    
                    # Multiselect für Kurse
                    selected_courses = st.multiselect(
                        "Kurse auswählen:",
                        options=sorted_courses,
                        help="Wählen Sie einen oder mehrere Kurse aus, um Schüler zu finden, die alle diese Kurse belegen."
                    )
                    
                    if selected_courses:
                        # Suche nach Schülern, die alle ausgewählten Kurse belegen
                        matched_students = search_students_by_courses(students_df, selected_courses)
                        
                        if not matched_students.empty:
                            st.success(f"{len(matched_students)} Schüler belegen alle ausgewählten Kurse.")
                            
                            # Ergebnisse anzeigen
                            for _, student in matched_students.iterrows():
                                with st.expander(f"{student['Name']} - {len(student['Kurse_Liste'])} Kurse"):
                                    st.write(f"**ID:** {student['student_id']}")
                                    st.write("**Kurse:**")
                                    
                                    # Hervorheben der ausgewählten Kurse
                                    for course in student['Kurse_Liste']:
                                        if course in selected_courses:
                                            st.markdown(f"- **:blue[{course}]**")
                                        else:
                                            st.write(f"- {course}")
                        else:
                            st.info("Keine Schüler gefunden, die alle ausgewählten Kurse belegen.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Tab 3: Personen-Heatmap
                with search_tabs[2]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Personen-Heatmap")
                    
                    # Option zum Filtern der Anzahl der Personen
                    max_students = st.slider("Maximale Anzahl an Schülern für die Heatmap:", 
                                             min_value=10, 
                                             max_value=min(100, len(students_df)), 
                                             value=min(30, len(students_df)),
                                             help="Bei vielen Schülern kann die Heatmap unübersichtlich werden.")
                    
                    # Filtern der Schüler basierend auf der Kursanzahl
                    students_with_course_count = students_df.copy()
                    students_with_course_count['course_count'] = students_with_course_count['Kurse_Liste'].apply(len)
                    filtered_students = students_with_course_count.nlargest(max_students, 'course_count')
                    
                    # Überschneidungsmatrix berechnen
                    overlap_matrix = calculate_student_overlap(filtered_students)
                    
                    if not overlap_matrix.empty:
                        # Tabs für verschiedene Visualisierungsoptionen
                        viz_tabs = st.tabs(["Interaktive Heatmap", "SVG-Export"])
                        
                        # Interaktive Heatmap mit Plotly
                        with viz_tabs[0]:
                            fig = plot_student_overlap_heatmap(overlap_matrix)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Export-Optionen
                                export_col1, export_col2 = st.columns(2)
                                with export_col1:
                                    if st.button("Als PNG exportieren", key="export_heatmap_png"):
                                        img_bytes = export_plotly_figure(fig, format="png")
                                        if img_bytes:
                                            st.download_button(
                                                label="PNG-Datei herunterladen",
                                                data=img_bytes,
                                                file_name="personen_heatmap.png",
                                                mime="image/png",
                                                key="download_heatmap_png"
                                            )
                                        else:
                                            st.error("Fehler beim Exportieren der Grafik als PNG.")
                                
                                with export_col2:
                                    if st.button("Als SVG exportieren", key="export_heatmap_svg_plotly"):
                                        img_bytes = export_plotly_figure(fig, format="svg")
                                        if img_bytes:
                                            st.download_button(
                                                label="SVG-Datei herunterladen",
                                                data=img_bytes,
                                                file_name="personen_heatmap_interaktiv.svg",
                                                mime="image/svg+xml",
                                                key="download_heatmap_svg_plotly"
                                            )
                                        else:
                                            st.error("Fehler beim Exportieren der Grafik als SVG.")
                            else:
                                st.error("Fehler bei der Erstellung der Heatmap.")
                        
                        # SVG-Export mit Matplotlib/Seaborn
                        with viz_tabs[1]:
                            svg_data = generate_student_overlap_svg(overlap_matrix)
                            if svg_data:
                                # SVG anzeigen
                                st.components.v1.html(svg_data, height=700)
                                
                                # Download-Button für SVG
                                download_btn = st.download_button(
                                    label="SVG-Datei herunterladen",
                                    data=svg_data,
                                    file_name="personen_heatmap.svg",
                                    mime="image/svg+xml"
                                )
                            else:
                                st.error("Fehler beim Generieren der SVG-Datei.")
                    else:
                        st.warning("Keine ausreichenden Daten für die Personen-Heatmap verfügbar.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Tab 4: Schülernetzwerk
                with search_tabs[3]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Schülernetzwerk")
                    
                    # Parameter für das Netzwerk
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        min_shared_courses = st.slider(
                            "Mindestanzahl gemeinsamer Kurse für eine Verbindung:",
                            min_value=1,
                            max_value=10,
                            value=2
                        )
                    
                    with col2:
                        # Dropdown für Fokusschüler
                        focus_options = ["Alle Schüler anzeigen"] + students_df['Name'].tolist()
                        focus_student = st.selectbox(
                            "Fokusschüler auswählen:",
                            options=focus_options
                        )
                    
                    # Netzwerkvisualisierung erstellen
                    focus_person = None if focus_student == "Alle Schüler anzeigen" else focus_student
                    
                    network_fig = create_person_network_graph(
                        students_df, 
                        focus_person=focus_person,
                        min_shared_courses=min_shared_courses
                    )
                    
                    if network_fig:
                        st.plotly_chart(network_fig, use_container_width=True)
                        
                        # Export-Optionen
                        export_col1, export_col2 = st.columns(2)
                        with export_col1:
                            if st.button("Als PNG exportieren", key="export_network_png"):
                                img_bytes = export_plotly_figure(network_fig, format="png", width=1200, height=800)
                                if img_bytes:
                                    st.download_button(
                                        label="PNG-Datei herunterladen",
                                        data=img_bytes,
                                        file_name="schueler_netzwerk.png",
                                        mime="image/png",
                                        key="download_network_png"
                                    )
                                else:
                                    st.error("Fehler beim Exportieren des Netzwerks als PNG.")
                        
                        with export_col2:
                            if st.button("Als SVG exportieren", key="export_network_svg"):
                                img_bytes = export_plotly_figure(network_fig, format="svg", width=1200, height=800)
                                if img_bytes:
                                    st.download_button(
                                        label="SVG-Datei herunterladen",
                                        data=img_bytes,
                                        file_name="schueler_netzwerk.svg",
                                        mime="image/svg+xml",
                                        key="download_network_svg"
                                    )
                                else:
                                    st.error("Fehler beim Exportieren des Netzwerks als SVG.")
                    else:
                        st.warning("Netzwerkgraph konnte nicht erstellt werden.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Keine Schülerdaten verfügbar.")
        
        except Exception as e:
            st.error(f"Fehler beim Laden der Schülerdaten: {str(e)}")
            logger.error(f"Fehler beim Laden der Schülerdaten: {e}")
            st.code(traceback.format_exc())
    
    # Tab 5: Erweiterte Visualisierung
    with tab5:
        st.markdown('<div class="sub-header">🔄 Erweiterte Visualisierung & Analyse</div>', unsafe_allow_html=True)
        
        # Alle Features in Tabs organisieren
        adv_tabs = st.tabs([
            "🔀 Sankey-Diagramm", 
            "🌐 3D-Netzwerk", 
            "📊 Cluster-Analyse", 
            "📑 PDF-Export",
            "🔐 DSGVO & Sicherheit"
        ])
        
        # Tab 1: Sankey-Diagramm
        with adv_tabs[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Schülerströme zwischen Kursen")
            
            try:
                # Schülerdaten laden
                with open('json/students.json', 'r', encoding='utf-8') as file:
                    students_data = json.load(file)
                
                # DataFrame erstellen
                students_df = pd.DataFrame(students_data)
                
                # Spaltenname standardisieren (groß schreiben)
                students_df.rename(columns={'name': 'Name'}, inplace=True)
                
                # Kursliste als separate Spalte für einfacheren Zugriff
                students_df['Kurse_Liste'] = students_df['courses'].apply(lambda courses: [course.strip() for course in str(courses).split(',')])
                
                # Parameter für das Sankey-Diagramm
                col1, col2 = st.columns(2)
                
                with col1:
                    max_students = st.slider(
                        "Maximale Anzahl an Schülern für die Analyse:",
                        min_value=20,
                        max_value=min(200, len(students_df)),
                        value=min(100, len(students_df))
                    )
                
                with col2:
                    min_shared = st.slider(
                        "Mindestanzahl gemeinsamer Schüler für eine Verbindung:",
                        min_value=1,
                        max_value=20,
                        value=3
                    )
                
                # Sankey-Diagramm erstellen
                if st.button("Sankey-Diagramm generieren", key="sankey_btn"):
                    with st.spinner("Erstelle Sankey-Diagramm..."):
                        sankey_fig = create_student_course_sankey(
                            students_df, 
                            max_students=max_students, 
                            min_shared_courses=min_shared
                        )
                        
                        if sankey_fig:
                            st.plotly_chart(sankey_fig, use_container_width=True)
                            
                            # Erklärung des Diagramms
                            st.info("""
                            **Über das Sankey-Diagramm:**
                            
                            Dieses Diagramm visualisiert die "Ströme" von Schülern zwischen verschiedenen Kursen. 
                            Die Breite der Verbindungen zeigt, wie viele Schüler beide Kurse gemeinsam belegen.
                            Je dicker die Verbindung, desto mehr Schüler teilen sich diese Kurskombination.
                            """)
                        else:
                            st.error("Konnte kein Sankey-Diagramm erstellen. Bitte Parameter anpassen.")
                else:
                    st.info("Klicke auf den Button, um das Sankey-Diagramm zu generieren.")
                
            except Exception as e:
                st.error(f"Fehler beim Erstellen des Sankey-Diagramms: {str(e)}")
                logger.error(f"Fehler beim Erstellen des Sankey-Diagramms: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: 3D-Netzwerk
        with adv_tabs[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("3D-Visualisierung des Schülernetzwerks")
            
            try:
                # Schülerdaten laden (falls noch nicht geladen)
                if 'students_df' not in locals():
                    with open('json/students.json', 'r', encoding='utf-8') as file:
                        students_data = json.load(file)
                    
                    # DataFrame erstellen
                    students_df = pd.DataFrame(students_data)
                    
                    # Spaltenname standardisieren (groß schreiben)
                    students_df.rename(columns={'name': 'Name'}, inplace=True)
                    
                    # Kursliste als separate Spalte für einfacheren Zugriff
                    students_df['Kurse_Liste'] = students_df['courses'].apply(lambda courses: [course.strip() for course in str(courses).split(',')])
                
                # Parameter für das 3D-Netzwerk
                col1, col2 = st.columns(2)
                
                with col1:
                    max_students_3d = st.slider(
                        "Maximale Anzahl an Schülern für das 3D-Netzwerk:",
                        min_value=20,
                        max_value=min(150, len(students_df)),
                        value=min(80, len(students_df)),
                        key="max_students_3d"
                    )
                
                with col2:
                    min_shared_3d = st.slider(
                        "Mindestanzahl gemeinsamer Kurse für eine Verbindung:",
                        min_value=1,
                        max_value=10,
                        value=2,
                        key="min_shared_3d"
                    )
                
                # 3D-Netzwerk erstellen
                if st.button("3D-Netzwerk generieren", key="network_3d_btn"):
                    with st.spinner("Erstelle 3D-Netzwerk..."):
                        network_3d_fig = create_3d_student_network(
                            students_df, 
                            max_students=max_students_3d, 
                            min_shared_courses=min_shared_3d
                        )
                        
                        if network_3d_fig:
                            st.plotly_chart(network_3d_fig, use_container_width=True)
                            
                            # Tipps zur Interaktion
                            st.info("""
                            **Interaktive 3D-Visualisierung:**
                            
                            - **Drehen:** Klicken und ziehen in der Visualisierung
                            - **Zoomen:** Mausrad oder Zwei-Finger-Geste
                            - **Details:** Hover über Knoten oder Verbindungen für mehr Informationen
                            - **Farben:** Zeigen die Anzahl der Kurse pro Schüler an
                            - **Knotengrößen:** Repräsentieren die Anzahl der Verbindungen
                            """)
                        else:
                            st.error("Konnte kein 3D-Netzwerk erstellen. Bitte Parameter anpassen.")
                else:
                    st.info("Klicke auf den Button, um das 3D-Netzwerk zu generieren.")
                
            except Exception as e:
                st.error(f"Fehler beim Erstellen des 3D-Netzwerks: {str(e)}")
                logger.error(f"Fehler beim Erstellen des 3D-Netzwerks: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 3: Cluster-Analyse
        with adv_tabs[2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Cluster-Analyse der Schülergruppen")
            
            try:
                # Schülerdaten laden (falls noch nicht geladen)
                if 'students_df' not in locals():
                    with open('json/students.json', 'r', encoding='utf-8') as file:
                        students_data = json.load(file)
                    
                    # DataFrame erstellen
                    students_df = pd.DataFrame(students_data)
                    
                    # Spaltenname standardisieren (groß schreiben)
                    students_df.rename(columns={'name': 'Name'}, inplace=True)
                    
                    # Kursliste als separate Spalte für einfacheren Zugriff
                    students_df['Kurse_Liste'] = students_df['courses'].apply(lambda courses: [course.strip() for course in str(courses).split(',')])
                
                # Parameter für die Cluster-Analyse
                col1, col2 = st.columns(2)
                
                with col1:
                    n_clusters = st.slider(
                        "Anzahl der Cluster:",
                        min_value=2,
                        max_value=10,
                        value=5
                    )
                
                with col2:
                    algorithm = st.selectbox(
                        "Clustering-Algorithmus:",
                        options=["kmeans", "hierarchical", "dbscan"],
                        index=0
                    )
                
                # Cluster-Analyse durchführen
                if st.button("Cluster-Analyse durchführen", key="cluster_btn"):
                    with st.spinner("Führe Cluster-Analyse durch..."):
                        clustered_df, cluster_fig = perform_student_clustering(
                            students_df, 
                            n_clusters=n_clusters, 
                            algorithm=algorithm
                        )
                        
                        if cluster_fig:
                            # Visualisierung anzeigen
                            st.plotly_chart(cluster_fig, use_container_width=True)
                            
                            # Zusammenfassung der Cluster
                            if clustered_df is not None:
                                # Cluster-Größen
                                cluster_sizes = clustered_df['Cluster'].value_counts().sort_index()
                                
                                # Durchschnittliche Kursanzahl pro Cluster
                                cluster_course_counts = clustered_df.groupby('Cluster')['Kurse_Liste'].apply(
                                    lambda x: sum(len(courses) for courses in x) / len(x)
                                ).round(2)
                                
                                # Tabelle mit Cluster-Infos
                                cluster_info = pd.DataFrame({
                                    'Cluster': cluster_sizes.index,
                                    'Anzahl Schüler': cluster_sizes.values,
                                    'Durchschnittliche Kursanzahl': [cluster_course_counts[i] for i in cluster_sizes.index]
                                })
                                
                                st.subheader("Cluster-Übersicht")
                                st.dataframe(cluster_info, use_container_width=True)
                                
                                # Detaillierte Informationen pro Cluster
                                st.subheader("Cluster-Details")
                                
                                for cluster_id in cluster_sizes.index:
                                    with st.expander(f"Cluster {cluster_id} ({cluster_sizes[cluster_id]} Schüler)"):
                                        cluster_students = clustered_df[clustered_df['Cluster'] == cluster_id]
                                        
                                        # Häufigste Kurse in diesem Cluster
                                        all_courses = []
                                        for courses in cluster_students['Kurse_Liste']:
                                            all_courses.extend(courses)
                                        
                                        course_counts = pd.Series(all_courses).value_counts()
                                        top_courses = course_counts.head(10)
                                        
                                        # Kurs-Verteilung visualisieren
                                        course_fig = px.bar(
                                            x=top_courses.index, 
                                            y=top_courses.values,
                                            labels={'x': 'Kurs', 'y': 'Anzahl Schüler'},
                                            title=f'Häufigste Kurse in Cluster {cluster_id}'
                                        )
                                        st.plotly_chart(course_fig, use_container_width=True)
                                        
                                        # Schüler in diesem Cluster auflisten
                                        st.write("**Schüler in diesem Cluster:**")
                                        st.dataframe(
                                            cluster_students[['Name', 'student_id']].head(20),
                                            use_container_width=True
                                        )
                                        if len(cluster_students) > 20:
                                            st.write(f"... und {len(cluster_students) - 20} weitere Schüler")
                        else:
                            st.error("Konnte keine Cluster-Analyse durchführen. Bitte Parameter anpassen.")
                else:
                    st.info("Klicke auf den Button, um die Cluster-Analyse zu starten.")
                
            except Exception as e:
                st.error(f"Fehler bei der Cluster-Analyse: {str(e)}")
                logger.error(f"Fehler bei der Cluster-Analyse: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 4: PDF-Export
        with adv_tabs[3]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("PDF-Bericht generieren")
            
            try:
                # Schülerdaten laden (falls noch nicht geladen)
                if 'students_df' not in locals():
                    with open('json/students.json', 'r', encoding='utf-8') as file:
                        students_data = json.load(file)
                    
                    # DataFrame erstellen
                    students_df = pd.DataFrame(students_data)
                    
                    # Spaltenname standardisieren (groß schreiben)
                    students_df.rename(columns={'name': 'Name'}, inplace=True)
                    
                    # Kursliste als separate Spalte für einfacheren Zugriff
                    students_df['Kurse_Liste'] = students_df['courses'].apply(lambda courses: [course.strip() for course in str(courses).split(',')])
                
                # Kursdaten laden
                with open('json/courses.json', 'r', encoding='utf-8') as file:
                    courses_data = json.load(file)
                
                courses_df = pd.DataFrame(courses_data)
                
                # Optionen für den PDF-Bericht
                st.write("### Berichtsoptionen")
                
                # Berichtstitel
                report_title = st.text_input(
                    "Berichtstitel:",
                    value="Kursplan-Analyse: Detaillierter Bericht"
                )
                
                # Zu inkludierende Visualisierungen
                st.write("**Einzuschließende Visualisierungen:**")
                
                include_heatmap = st.checkbox("Schüler-Kursüberschneidungen (Heatmap)", value=True)
                include_sankey = st.checkbox("Schülerströme zwischen Kursen (Sankey)", value=True)
                include_network = st.checkbox("Schülernetzwerk (2D-Visualisierung)", value=True)
                include_clusters = st.checkbox("Cluster-Analyse", value=False)
                
                # PDF generieren
                if st.button("PDF-Bericht generieren", key="pdf_btn"):
                    with st.spinner("Erstelle PDF-Bericht..."):
                        # Visualisierungen erstellen, die in den Bericht aufgenommen werden sollen
                        charts = []
                        
                        # Überschneidungsmatrix für verschiedene Visualisierungen
                        overlap_matrix = calculate_student_overlap(students_df)
                        
                        # Heatmap hinzufügen
                        if include_heatmap:
                            try:
                                heatmap_fig = plot_student_overlap_heatmap(overlap_matrix)
                                if heatmap_fig:
                                    # Heatmap als Bild speichern
                                    img_bytes = heatmap_fig.to_image(format="png", width=800, height=600, scale=2)
                                    charts.append(img_bytes)
                            except Exception as viz_error:
                                logger.error(f"Fehler beim Erstellen der Heatmap: {viz_error}")
                        
                        # Sankey-Diagramm hinzufügen
                        if include_sankey:
                            try:
                                sankey_fig = create_student_course_sankey(
                                    students_df, 
                                    max_students=100, 
                                    min_shared_courses=3
                                )
                                if sankey_fig:
                                    # Sankey als Bild speichern
                                    img_bytes = sankey_fig.to_image(format="png", width=800, height=600, scale=2)
                                    charts.append(img_bytes)
                            except Exception as viz_error:
                                logger.error(f"Fehler beim Erstellen des Sankey-Diagramms: {viz_error}")
                        
                        # Netzwerk hinzufügen
                        if include_network:
                            try:
                                network_fig = create_person_network_graph(
                                    students_df, 
                                    focus_person=None,
                                    min_shared_courses=2
                                )
                                if network_fig:
                                    # Netzwerk als Bild speichern
                                    img_bytes = network_fig.to_image(format="png", width=800, height=600, scale=2)
                                    charts.append(img_bytes)
                            except Exception as viz_error:
                                logger.error(f"Fehler beim Erstellen des Netzwerks: {viz_error}")
                        
                        # Cluster-Analyse hinzufügen
                        if include_clusters:
                            try:
                                _, cluster_fig = perform_student_clustering(
                                    students_df, 
                                    n_clusters=5, 
                                    algorithm="kmeans"
                                )
                                if cluster_fig:
                                    # Cluster-Visualisierung als Bild speichern
                                    img_bytes = cluster_fig.to_image(format="png", width=800, height=600, scale=2)
                                    charts.append(img_bytes)
                            except Exception as viz_error:
                                logger.error(f"Fehler beim Erstellen der Cluster-Visualisierung: {viz_error}")
                        
                        # PDF generieren
                        pdf_bytes = generate_pdf_report(
                            students_df,
                            courses_df,
                            overlap_matrix=overlap_matrix,
                            charts=charts,
                            title=report_title
                        )
                        
                        if pdf_bytes:
                            # Download-Button für PDF
                            st.download_button(
                                label="PDF-Bericht herunterladen",
                                data=pdf_bytes,
                                file_name="kursplan_analyse_bericht.pdf",
                                mime="application/pdf"
                            )
                            
                            st.success("PDF-Bericht wurde erfolgreich erstellt! Klicke auf den Button oben, um ihn herunterzuladen.")
                        else:
                            st.error("Fehler beim Erstellen des PDF-Berichts.")
                else:
                    st.info("Wähle die gewünschten Optionen und klicke auf den Button, um den PDF-Bericht zu generieren.")
                
            except Exception as e:
                st.error(f"Fehler beim Erstellen des PDF-Berichts: {str(e)}")
                logger.error(f"Fehler beim Erstellen des PDF-Berichts: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 5: DSGVO & Sicherheit
        with adv_tabs[4]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("DSGVO-Compliance & Datensicherheit")
            
            try:
                # Schülerdaten laden (falls noch nicht geladen)
                if 'students_df' not in locals():
                    with open('json/students.json', 'r', encoding='utf-8') as file:
                        students_data = json.load(file)
                    
                    # DataFrame erstellen
                    students_df = pd.DataFrame(students_data)
                    
                    # Spaltenname standardisieren (groß schreiben)
                    students_df.rename(columns={'name': 'Name'}, inplace=True)
                    
                    # Kursliste als separate Spalte für einfacheren Zugriff
                    students_df['Kurse_Liste'] = students_df['courses'].apply(lambda courses: [course.strip() for course in str(courses).split(',')])
                
                # DSGVO-Tabs
                dsgvo_tabs = st.tabs([
                    "Daten anonymisieren", 
                    "QR-Code-Generator", 
                    "Audit-Logs", 
                    "Datenschutzerklärung"
                ])
                
                # Daten anonymisieren
                with dsgvo_tabs[0]:
                    st.write("### Datensatz anonymisieren")
                    st.write("""
                    Diese Funktion ermöglicht es, personenbezogene Daten gemäß DSGVO-Anforderungen zu anonymisieren,
                    bevor sie für Analysen oder Exporte verwendet werden.
                    """)
                    
                    # Anonymisierungslevel auswählen
                    anon_level = st.radio(
                        "Anonymisierungslevel:",
                        options=["low", "medium", "high"],
                        index=1,
                        help="Low: Nur Teile der Namen werden anonymisiert. Medium: Namen weitgehend anonymisiert. High: Vollständige Anonymisierung aller personenbezogenen Daten."
                    )
                    
                    if st.button("Daten anonymisieren", key="anon_btn"):
                        with st.spinner("Anonymisiere Daten..."):
                            anon_df, anon_info = anonymize_student_data(
                                students_df,
                                anonymization_level=anon_level
                            )
                            
                            if anon_df is not None:
                                # Anonymisierte Daten anzeigen
                                st.subheader("Anonymisierte Daten")
                                st.dataframe(anon_df.head(20), use_container_width=True)
                                
                                # Anonymisierungs-Info anzeigen
                                st.subheader("Anonymisierungs-Details")
                                st.write(f"- **Originalanzahl Datensätze:** {anon_info['original_count']}")
                                st.write(f"- **Anonymisierte Datensätze:** {anon_info['anonymized_count']}")
                                st.write(f"- **Anonymisierungslevel:** {anon_info['anonymization_level']}")
                                st.write(f"- **Zeitstempel:** {anon_info['timestamp']}")
                                
                                # Download-Button für anonymisierte Daten
                                csv_buffer = BytesIO()
                                anon_df.to_csv(csv_buffer, index=False)
                                csv_buffer.seek(0)
                                
                                st.download_button(
                                    label="Anonymisierte Daten herunterladen (CSV)",
                                    data=csv_buffer,
                                    file_name=f"anonymisierte_daten_{anon_level}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("Fehler bei der Anonymisierung der Daten.")
                
                # QR-Code-Generator
                with dsgvo_tabs[1]:
                    st.write("### QR-Code Generator für Schülerdaten")
                    st.write("""
                    Erstellen Sie QR-Codes für den schnellen und sicheren Zugriff auf Schülerdaten.
                    Diese können für Identifikation bei Veranstaltungen oder sichere Datenübertragung verwendet werden.
                    """)
                    
                    # Schüler auswählen
                    selected_student = st.selectbox(
                        "Schüler auswählen:",
                        options=students_df['Name'].tolist(),
                        index=0
                    )
                    
                    # Datenschutzoptionen
                    include_id = st.checkbox("Schüler-ID einschließen", value=True)
                    include_courses = st.checkbox("Kursliste einschließen", value=True)
                    
                    if st.button("QR-Code generieren", key="qr_btn"):
                        with st.spinner("Generiere QR-Code..."):
                            # Schülerdaten abrufen
                            student_data = students_df[students_df['Name'] == selected_student].iloc[0].to_dict()
                            
                            # Daten für QR-Code vorbereiten
                            qr_data = {
                                "name": student_data['Name']
                            }
                            
                            if include_id:
                                qr_data["id"] = str(student_data['student_id'])
                            
                            if include_courses:
                                qr_data["courses"] = student_data['Kurse_Liste']
                            
                            # QR-Code-Optionen
                            qr_col1, qr_col2 = st.columns(2)
                            
                            with qr_col1:
                                qr_fill_color = st.color_picker("QR-Code Farbe", "#000000")
                                
                            with qr_col2:
                                qr_back_color = st.color_picker("Hintergrund Farbe", "#FFFFFF")
                            
                            # Logo-Option
                            use_logo = st.checkbox("Logo im QR-Code anzeigen", value=False)
                            logo_path = None
                            
                            if use_logo:
                                logo_upload = st.file_uploader("Logo-Datei hochladen (empfohlen: quadratisches PNG)", type=["png", "jpg", "jpeg"])
                                if logo_upload is not None:
                                    # Temporäre Datei speichern
                                    logo_path = f"temp_logo_{uuid.uuid4()}.png"
                                    with open(logo_path, "wb") as f:
                                        f.write(logo_upload.getbuffer())
                            
                            # QR-Code generieren mit verbesserten Optionen
                            qr_buffer = generate_qr_code(
                                data=qr_data, 
                                name=selected_student,
                                size=300,  # Größerer QR-Code
                                logo_path=logo_path,
                                fill_color=qr_fill_color,
                                back_color=qr_back_color
                            )
                            
                            if qr_buffer:
                                # QR-Code in attraktiver Karte anzeigen
                                st.markdown("""
                                <style>
                                .qr-card {
                                    background-color: white;
                                    border-radius: 10px;
                                    padding: 20px;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                    margin-bottom: 20px;
                                    text-align: center;
                                }
                                .qr-title {
                                    font-size: 1.2em;
                                    font-weight: bold;
                                    margin-bottom: 10px;
                                }
                                </style>
                                <div class="qr-card">
                                    <div class="qr-title">QR-Code für {}</div>
                                </div>
                                """.format(selected_student), unsafe_allow_html=True)
                                
                                st.image(qr_buffer, width=300)
                                
                                # Informationen zum QR-Code
                                st.info(f"""
                                Der QR-Code enthält folgende Informationen:
                                - Name: {selected_student}
                                {f"- ID: {qr_data['id']}" if 'id' in qr_data else ""}
                                {f"- Kurse: {len(qr_data['courses'])}" if 'courses' in qr_data else ""}
                                
                                Sie können diesen QR-Code scannen, um schnell auf die Schülerdaten zuzugreifen.
                                """)
                                
                                # Download-Button für QR-Code mit verbesserten Optionen
                                qr_buffer.seek(0)
                                st.download_button(
                                    label="QR-Code herunterladen",
                                    data=qr_buffer,
                                    file_name=f"qrcode_{selected_student.replace(' ', '_').replace(',', '')}.png",
                                    mime="image/png",
                                    help="Laden Sie den QR-Code als PNG-Datei herunter"
                                )
                            else:
                                st.error("Fehler bei der Generierung des QR-Codes.")
                
                # Audit-Logs
                with dsgvo_tabs[2]:
                    st.write("### Audit-Logs einsehen")
                    st.write("""
                    Gemäß DSGVO müssen alle Datenoperationen protokolliert werden. 
                    Hier können Sie die Audit-Logs einsehen und filtern.
                    """)
                    
                    # Audit-Logger initialisieren
                    audit_logger = AuditLogger()
                    
                    # Beispielhafte Protokollierung dieser Ansicht
                    audit_logger.log_action(
                        action="view",
                        data_type="audit_logs",
                        details={"page": "Erweiterte Visualisierung", "tab": "DSGVO & Sicherheit"},
                        user="admin"  # In einer echten Anwendung würde hier der angemeldete Benutzer stehen
                    )
                    
                    # Filter für Logs
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        filter_action = st.selectbox(
                            "Aktion:",
                            options=["Alle", "view", "export", "anonymize", "session_end"],
                            index=0
                        )
                    
                    with col2:
                        filter_data_type = st.selectbox(
                            "Datentyp:",
                            options=["Alle", "student", "course", "audit_logs", "session"],
                            index=0
                        )
                    
                    with col3:
                        filter_user = st.selectbox(
                            "Benutzer:",
                            options=["Alle", "admin", "anonymous"],
                            index=0
                        )
                    
                    # Logs abrufen
                    logs = audit_logger.get_logs(
                        filter_action=None if filter_action == "Alle" else filter_action,
                        filter_data_type=None if filter_data_type == "Alle" else filter_data_type,
                        filter_user=None if filter_user == "Alle" else filter_user
                    )
                    
                    # Logs anzeigen
                    if logs:
                        # DataFrame für bessere Darstellung
                        log_df = pd.DataFrame(logs)
                        st.dataframe(log_df, use_container_width=True)
                        
                        # Download-Button für Logs
                        csv_buffer = BytesIO()
                        log_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        st.download_button(
                            label="Audit-Logs exportieren (CSV)",
                            data=csv_buffer,
                            file_name="audit_logs_export.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Keine Audit-Logs gefunden, die den Filterkriterien entsprechen.")
                
                # Datenschutzerklärung
                with dsgvo_tabs[3]:
                    st.write("### Datenschutzerklärung und DSGVO-Informationen")
                    
                    st.markdown("""
                    ## Datenschutzerklärung für Kursplan-Analyse-Tool
                    
                    **Verantwortlich für die Datenverarbeitung:**
                    
                    Bildungseinrichtung XYZ  
                    Musterstraße 123  
                    12345 Musterstadt  
                    datenschutz@bildungseinrichtung-xyz.de
                    
                    ### 1. Art der verarbeiteten Daten
                    
                    Diese Anwendung verarbeitet folgende personenbezogene Daten:
                    
                    - Namen von Schülern
                    - Schüler-IDs
                    - Kursbelegungsinformationen
                    - Indirekt: Beziehungen zwischen Schülern durch gemeinsame Kurse
                    
                    ### 2. Zweck der Datenverarbeitung
                    
                    Die Datenverarbeitung erfolgt zu folgenden Zwecken:
                    
                    - Analyse der Kurs- und Stundenplanstruktur
                    - Optimierung der Kursplanung
                    - Erkennung von Überschneidungen und Mustern
                    - Erstellung von Berichten für Bildungsplanungszwecke
                    
                    ### 3. Rechtsgrundlage
                    
                    Die Verarbeitung basiert auf folgenden Rechtsgrundlagen:
                    
                    - Einwilligung der betroffenen Personen (Art. 6 Abs. 1 lit. a DSGVO)
                    - Erfüllung einer rechtlichen Verpflichtung (Art. 6 Abs. 1 lit. c DSGVO)
                    - Berechtigtes Interesse an effektiver Bildungsplanung (Art. 6 Abs. 1 lit. f DSGVO)
                    
                    ### 4. Datensicherheit
                    
                    Zum Schutz der verarbeiteten Daten werden folgende Maßnahmen ergriffen:
                    
                    - Verschlüsselung bei der Übertragung
                    - Zugriffsbeschränkungen für autorisierte Personen
                    - Anonymisierungsoptionen für Berichte und Exporte
                    - Regelmäßige Sicherheitsüberprüfungen
                    - Audit-Logging aller Datenoperationen
                    
                    ### 5. Betroffenenrechte
                    
                    Betroffene Personen haben folgende Rechte:
                    
                    - Recht auf Auskunft
                    - Recht auf Berichtigung
                    - Recht auf Löschung
                    - Recht auf Einschränkung der Verarbeitung
                    - Recht auf Datenübertragbarkeit
                    - Widerspruchsrecht
                    
                    ### 6. Kontakt zum Datenschutzbeauftragten
                    
                    Bei Fragen zum Datenschutz:
                    
                    Datenschutzbeauftragter  
                    Bildungseinrichtung XYZ  
                    datenschutz@bildungseinrichtung-xyz.de  
                    Tel.: +49 123 4567890
                    
                    ### 7. Aktualisierung
                    
                    Diese Datenschutzerklärung wurde zuletzt am 31.08.2025 aktualisiert.
                    """)
                    
                    # Einfaches DSGVO-Compliance-Tool
                    st.subheader("DSGVO-Compliance-Check")
                    
                    # Einfacher Compliance-Check
                    compliance_checks = {
                        "Daten werden nur für festgelegte Zwecke verwendet": True,
                        "Datensparsamkeit wird eingehalten": True,
                        "Betroffene wurden über die Datenverarbeitung informiert": False,
                        "Einwilligungen wurden eingeholt": False,
                        "Daten sind durch angemessene Maßnahmen geschützt": True,
                        "Löschkonzept für nicht mehr benötigte Daten existiert": False,
                        "Auftragsverarbeitungsverträge mit Dienstleistern bestehen": False,
                        "Verfahrensverzeichnis wurde erstellt": False,
                        "Datenschutz-Folgenabschätzung wurde durchgeführt": False
                    }
                    
                    # Compliance-Status anzeigen
                    compliance_count = sum(compliance_checks.values())
                    compliance_percent = (compliance_count / len(compliance_checks)) * 100
                    
                    st.progress(compliance_percent / 100)
                    st.write(f"DSGVO-Compliance: {compliance_percent:.1f}% ({compliance_count}/{len(compliance_checks)} Kriterien erfüllt)")
                    
                    for check, status in compliance_checks.items():
                        st.checkbox(check, value=status, disabled=True)
                    
                    st.warning("Dieses Tool ersetzt keine professionelle Datenschutzberatung. Bitte konsultieren Sie einen Datenschutzexperten für eine vollständige DSGVO-Compliance-Prüfung.")
            
            except Exception as e:
                st.error(f"Fehler im DSGVO & Sicherheits-Tab: {str(e)}")
                logger.error(f"Fehler im DSGVO & Sicherheits-Tab: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 6: Developer-Übersicht
    with tab5:
        st.markdown('<div class="sub-header">🛠️ Developer-Übersicht</div>', unsafe_allow_html=True)
        
        # Mehrere Abschnitte für verschiedene Entwicklerinformationen
        dev_tabs = st.tabs(["📂 Datenstruktur", "📈 Performance", "🔄 API", "📚 Dokumentation"])
        
        # Tab 1: Datenstruktur
        with dev_tabs[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("JSON-Datenstruktur")
            
            # Dateistruktur anzeigen
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**CSV-Dateien:**")
                try:
                    csv_files = [f for f in os.listdir('csv') if f.endswith('.csv')]
                    for file in csv_files:
                        with st.expander(file):
                            try:
                                df = pd.read_csv(f'csv/{file}', nrows=5)
                                st.dataframe(df, use_container_width=True)
                                st.caption(f"Spalten: {', '.join(df.columns)}")
                            except Exception as e:
                                st.error(f"Fehler beim Lesen der Datei: {str(e)}")
                except Exception as e:
                    st.error(f"Fehler beim Auflisten der CSV-Dateien: {str(e)}")
            
            with col2:
                st.write("**JSON-Dateien:**")
                try:
                    json_files = [f for f in os.listdir('json') if f.endswith('.json')]
                    for file in json_files:
                        with st.expander(file):
                            try:
                                with open(f'json/{file}', 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                                    # Stichprobe anzeigen
                                    if isinstance(data, list) and len(data) > 0:
                                        st.json(data[0])
                                        st.caption(f"Anzahl Einträge: {len(data)}")
                                    else:
                                        st.json(data)
                            except Exception as e:
                                st.error(f"Fehler beim Lesen der Datei: {str(e)}")
                except Exception as e:
                    st.error(f"Fehler beim Auflisten der JSON-Dateien: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: Performance
        with dev_tabs[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Performance-Metriken")
            
            # Speichernutzung anzeigen
            st.write("**Speichernutzung:**")
            
            def get_size(obj):
                """Ermittelt die Größe eines Objekts in MB"""
                try:
                    size_bytes = sys.getsizeof(obj)
                    return size_bytes / (1024 * 1024)  # MB
                except:
                    return 0
            
            # Beispiel für Performance-Metriken
            col1, col2, col3 = st.columns(3)
            
            try:
                # Daten laden für die Performanceanalyse
                with open('json/students.json', 'r', encoding='utf-8') as f:
                    students_data = json.load(f)
                with open('json/courses.json', 'r', encoding='utf-8') as f:
                    courses_data = json.load(f)
                with open('json/timetable.json', 'r', encoding='utf-8') as f:
                    timetable_data = json.load(f)
                
                # Metriken berechnen
                students_size = get_size(students_data)
                courses_size = get_size(courses_data)
                timetable_size = get_size(timetable_data)
                
                with col1:
                    st.metric("Schülerdaten", f"{students_size:.2f} MB")
                    st.metric("Anzahl Schüler", f"{len(students_data)}")
                
                with col2:
                    st.metric("Kursdaten", f"{courses_size:.2f} MB")
                    st.metric("Anzahl Kurse", f"{len(courses_data)}")
                
                with col3:
                    st.metric("Stundenplan", f"{timetable_size:.2f} MB")
                    st.metric("Gesamtgröße", f"{(students_size + courses_size + timetable_size):.2f} MB")
                
                # Performance-Diagramm
                st.write("**Datenmenge nach Dateiart:**")
                performance_data = pd.DataFrame({
                    'Datei': ['Schülerdaten', 'Kursdaten', 'Stundenplan'],
                    'Größe (MB)': [students_size, courses_size, timetable_size]
                })
                
                fig = px.bar(
                    performance_data, 
                    x='Datei', 
                    y='Größe (MB)',
                    color='Datei',
                    title="Speicherverbrauch nach Dateiart"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance-Test für Schülersuche
                st.write("**Performance-Test: Schülersuche**")
                
                if st.button("Performance-Test durchführen"):
                    with st.spinner("Test läuft..."):
                        # Studenten-DataFrame erstellen
                        students_df = pd.DataFrame(students_data)
                        
                        # Spaltenname standardisieren (groß schreiben)
                        students_df.rename(columns={'name': 'Name'}, inplace=True)
                        
                        students_df['Kurse_Liste'] = students_df['courses'].apply(lambda courses: [course.strip() for course in str(courses).split(',')])
                        
                        # Zeit messen für verschiedene Operationen
                        results = []
                        
                        # Test 1: Suche nach Namen
                        start_time = time.time()
                        search_students_by_name(students_df, "Müller")
                        name_search_time = time.time() - start_time
                        results.append({"Operation": "Namenssuche", "Zeit (s)": name_search_time})
                        
                        # Test 2: Suche nach Kursen
                        start_time = time.time()
                        search_students_by_courses(students_df, ["Mathematik"])
                        course_search_time = time.time() - start_time
                        results.append({"Operation": "Kurssuche", "Zeit (s)": course_search_time})
                        
                        # Test 3: Berechnung der Schülerüberschneidung
                        start_time = time.time()
                        calculate_student_overlap(students_df.head(30))
                        overlap_time = time.time() - start_time
                        results.append({"Operation": "Überschneidungsberechnung", "Zeit (s)": overlap_time})
                        
                        # Test 4: SVG-Generierung
                        start_time = time.time()
                        overlap_matrix = calculate_student_overlap(students_df.head(20))
                        generate_student_overlap_svg(overlap_matrix)
                        svg_time = time.time() - start_time
                        results.append({"Operation": "SVG-Generierung", "Zeit (s)": svg_time})
                        
                        # Ergebnisse anzeigen
                        results_df = pd.DataFrame(results)
                        
                        # Tabelle
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Balkendiagramm
                        fig = px.bar(
                            results_df, 
                            x='Operation', 
                            y='Zeit (s)',
                            color='Operation',
                            title="Performance-Vergleich verschiedener Operationen"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Fehler bei der Performance-Analyse: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 3: API
        with dev_tabs[2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("API-Endpoints")
            
            # Beispiel-API-Dokumentation
            st.write("""
            ## Integrationsschnittstellen
            
            Diese Anwendung kann über REST-API-Endpunkte angesprochen werden. Hier sind die verfügbaren Endpunkte:
            """)
            
            api_endpoints = [
                {
                    "Endpoint": "/api/students",
                    "Methode": "GET",
                    "Beschreibung": "Liste aller Schüler abrufen",
                    "Parameter": "limit (optional): Maximale Anzahl der zurückgegebenen Einträge"
                },
                {
                    "Endpoint": "/api/students/{id}",
                    "Methode": "GET",
                    "Beschreibung": "Details eines bestimmten Schülers abrufen",
                    "Parameter": "id: ID des Schülers"
                },
                {
                    "Endpoint": "/api/courses",
                    "Methode": "GET",
                    "Beschreibung": "Liste aller Kurse abrufen",
                    "Parameter": "type (optional): Kurstyp-Filter (z.B. 'Leistungskurs')"
                },
                {
                    "Endpoint": "/api/timetable/{student_id}",
                    "Methode": "GET",
                    "Beschreibung": "Stundenplan eines Schülers abrufen",
                    "Parameter": "student_id: ID des Schülers"
                }
            ]
            
            # API-Tabelle anzeigen
            st.table(pd.DataFrame(api_endpoints))
            
            # Beispiel-Nutzung
            st.write("### Beispiel-Nutzung")
            
            with st.expander("Python-Beispiel"):
                st.code("""
import requests

# Alle Schüler abrufen
response = requests.get('http://localhost:8501/api/students')
students = response.json()

# Stundenplan eines bestimmten Schülers abrufen
student_id = "S12345"
response = requests.get(f'http://localhost:8501/api/timetable/{student_id}')
timetable = response.json()

print(f"Gefundene Schüler: {len(students)}")
print(f"Stundenplan für Schüler {student_id}: {timetable}")
                """, language="python")
            
            with st.expander("JavaScript-Beispiel"):
                st.code("""
// Alle Kurse abrufen
fetch('http://localhost:8501/api/courses')
  .then(response => response.json())
  .then(courses => {
    console.log(`Gefundene Kurse: ${courses.length}`);
    // Weitere Verarbeitung...
  });

// Nur Leistungskurse abrufen
fetch('http://localhost:8501/api/courses?type=Leistungskurs')
  .then(response => response.json())
  .then(courses => {
    console.log(`Gefundene Leistungskurse: ${courses.length}`);
    // Weitere Verarbeitung...
  });
                """, language="javascript")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 4: Dokumentation
        with dev_tabs[3]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Technische Dokumentation")
            
            # Technologien
            st.write("### Verwendete Technologien")
            
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.write("**Frontend:**")
                st.markdown("""
                - Streamlit
                - Plotly
                - Matplotlib/Seaborn für SVG-Export
                - Pandas für Datenverarbeitung
                """)
            
            with tech_col2:
                st.write("**Backend/Daten:**")
                st.markdown("""
                - JSON-Dateispeicherung
                - Python-Datenverarbeitungslogik
                - NetworkX für Netzwerkvisualisierungen
                - REST-API-Schnittstellen
                """)
            
            # Architekturdiagramm
            st.write("### Architekturübersicht")
            
            # Architekturdiagramm mit Plotly
            fig = go.Figure()
            
            # Knoten definieren
            nodes = [
                {"name": "Frontend (Streamlit)", "x": 0, "y": 0},
                {"name": "Datenverarbeitung", "x": 0, "y": -1},
                {"name": "Visualisierung", "x": 1, "y": -0.5},
                {"name": "API-Layer", "x": -1, "y": -0.5},
                {"name": "JSON-Daten", "x": 0, "y": -2}
            ]
            
            # Kanten definieren
            edges = [
                {"from": 0, "to": 1},
                {"from": 1, "to": 2},
                {"from": 1, "to": 3},
                {"from": 1, "to": 4},
                {"from": 3, "to": 4}
            ]
            
            # Knoten zeichnen
            for node in nodes:
                fig.add_annotation(
                    x=node["x"],
                    y=node["y"],
                    text=node["name"],
                    showarrow=False,
                    font=dict(size=14, color="white"),
                    bgcolor="#2563EB",
                    bordercolor="#1E40AF",
                    borderwidth=2,
                    borderpad=4,
                    opacity=0.8
                )
            
            # Kanten zeichnen
            for edge in edges:
                fig.add_shape(
                    type="line",
                    x0=nodes[edge["from"]]["x"],
                    y0=nodes[edge["from"]]["y"],
                    x1=nodes[edge["to"]]["x"],
                    y1=nodes[edge["to"]]["y"],
                    line=dict(color="#94A3B8", width=2)
                )
            
            # Layout anpassen
            fig.update_layout(
                showlegend=False,
                xaxis=dict(visible=False, range=[-1.5, 1.5]),
                yaxis=dict(visible=False, range=[-2.5, 0.5]),
                plot_bgcolor="white",
                margin=dict(l=0, r=0, t=0, b=0),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Codebeispiele
            st.write("### Wichtige Codebeispiele")
            
            with st.expander("Schülersuche nach Namen"):
                st.code("""
def search_students_by_name(students_df, search_query):
    \"\"\"Sucht Schüler anhand ihres Namens\"\"\"
    if students_df.empty or not search_query:
        return pd.DataFrame()
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Groß-/Kleinschreibung ignorieren und mehrere Suchbegriffe unterstützen
        search_terms = search_query.lower().split()
        
        # Suche nach jedem Begriff im Namen
        matched_students = students_df.copy()
        for term in search_terms:
            matched_students = matched_students[matched_students['Name'].str.lower().str.contains(term)]
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Personensuche: {end_time - start_time:.2f} Sekunden")
        
        return matched_students
        
    except Exception as e:
        logger.error(f"Fehler bei der Personensuche: {e}")
        return pd.DataFrame()
                """, language="python")
            
            with st.expander("SVG-Heatmap-Generierung"):
                st.code("""
def generate_student_overlap_svg(overlap_matrix, title="Kursüberschneidungen zwischen Schülern"):
    \"\"\"Erstellt eine SVG-Datei der Schülerüberschneidungsheatmap mit Matplotlib (nicht-interaktiv)\"\"\"
    if overlap_matrix.empty:
        return None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Erstelle eine eigene Farbpalette für die Heatmap
        colors = ['#F3F4F6', '#DBEAFE', '#93C5FD', '#3B82F6', '#1D4ED8']
        custom_cmap = LinearSegmentedColormap.from_list('custom_YlGnBu', colors)
        
        # Größe der Figur basierend auf der Anzahl der Schüler anpassen
        n_students = len(overlap_matrix)
        figsize = (max(10, n_students/3), max(8, n_students/3))
        
        # Erstelle die Figur und die Heatmap
        plt.figure(figsize=figsize)
        
        # Erstelle die Heatmap
        heatmap = sns.heatmap(
            overlap_matrix,
            cmap=custom_cmap,
            annot=True,  # Werte anzeigen
            fmt="d",     # Ganzzahlen anzeigen
            linewidths=0.5,
            cbar_kws={"label": "Anzahl gemeinsamer Kurse"}
        )
        
        # Titel und Achsenbeschriftungen
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Schüler', fontsize=12, labelpad=10)
        plt.ylabel('Schüler', fontsize=12, labelpad=10)
        
        # X-Achsenbeschriftungen rotieren
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        
        # Layout optimieren
        plt.tight_layout()
        
        # In SVG-Datei speichern
        svg_buffer = io.BytesIO()
        plt.savefig(svg_buffer, format='svg', bbox_inches='tight')
        plt.close()
        
        # Buffer zurücksetzen und SVG-Daten zurückgeben
        svg_buffer.seek(0)
        svg_data = svg_buffer.getvalue().decode('utf-8')
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"SVG-Heatmap-Generierung: {end_time - start_time:.2f} Sekunden")
        
        return svg_data
        
    except Exception as e:
        logger.error(f"Fehler bei der SVG-Heatmap-Generierung: {e}")
        return None
                """, language="python")
            
            # Version und letzte Aktualisierung
            st.write("### Versions- und Aktualisierungsinformationen")
            
            version_col1, version_col2 = st.columns(2)
            
            with version_col1:
                st.metric("Version", VERSION)
                st.write(f"**Letzte Aktualisierung:** {datetime.now().strftime('%d.%m.%Y')}")
            
            with version_col2:
                st.write("**Changelog:**")
                st.markdown("""
                - Überschneidungsanalyse entfernt
                - Personensuche (Name, Kurse) hinzugefügt
                - Personen-Heatmap mit SVG-Export hinzugefügt
                - Schülernetzwerk-Visualisierung hinzugefügt
                - Developer-Übersicht hinzugefügt
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
if __name__ == "__main__":
    main()
