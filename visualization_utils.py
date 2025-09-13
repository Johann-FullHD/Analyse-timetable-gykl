import base64
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from io import BytesIO

# Datenvisualisierung
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
# QR-Code Generierung
import qrcode
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from reportlab.lib import colors
# PDF-Erstellung
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
# Datenanalyse und Clustering
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler

# Matplotlib für nicht-interaktive SVG-Export
matplotlib.use('Agg')

# Logging-Konfiguration
logger = logging.getLogger(__name__)

# Cache-Gültigkeit in Sekunden
CACHE_TTL = 3600

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

@st.cache_data(ttl=CACHE_TTL)
def calculate_course_student_avg(courses_df, students_df):
    """Berechnet den Durchschnitt der Schüler pro Kurs mit demografischen Details"""
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        if courses_df.empty or students_df.empty:
            return None, None
        
        # Durchschnittliche Anzahl der Schüler pro Kurs
        avg_students = courses_df['Teilnehmerzahl'].mean()
        
        # Demografische Aufschlüsselung, falls verfügbar
        demographics = {}
        if 'Geschlecht' in students_df.columns:
            demographics['Geschlecht'] = students_df['Geschlecht'].value_counts().to_dict()
        
        if 'Jahrgang' in students_df.columns:
            demographics['Jahrgang'] = students_df['Jahrgang'].value_counts().to_dict()
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Kurs-Schüler-Durchschnitt: {end_time - start_time:.2f} Sekunden")
        
        return avg_students, demographics
    except Exception as e:
        logger.error(f"Fehler bei der Berechnung des Kurs-Schüler-Durchschnitts: {e}")
        return None, None

def plot_course_student_counts(courses_df, highlight_course=None, max_courses=20):
    """Erstellt ein Balkendiagramm der Schülerzahlen pro Kurs.
    
    Args:
        courses_df (pd.DataFrame): DataFrame mit Kursinformationen
        highlight_course (str, optional): Kurs, der hervorgehoben werden soll
        max_courses (int, optional): Maximale Anzahl von Kursen, die angezeigt werden sollen
        
    Returns:
        matplotlib.figure.Figure or None: Das erstellte Diagramm oder None bei Fehler
    """
    if courses_df is None or courses_df.empty:
        logger.warning("Keine Kursdaten zum Visualisieren vorhanden")
        # Leeres Diagramm mit Hinweis erstellen
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Keine Kursdaten vorhanden", 
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Prüfen, ob erforderliche Spalten vorhanden sind
        required_cols = ['Kurs', 'Teilnehmerzahl']
        if not all(col in courses_df.columns for col in required_cols):
            logger.error(f"Erforderliche Spalten fehlen. Vorhanden: {courses_df.columns.tolist()}")
            # Fehlermeldung im Diagramm
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Fehler: Erforderliche Spalten fehlen in den Kursdaten", 
                    ha='center', va='center', fontsize=14, color='red')
            ax.set_axis_off()
            return fig
        
        # Top-Kurse nach Teilnehmerzahl
        top_courses = courses_df.sort_values('Teilnehmerzahl', ascending=False).head(max_courses)
        
        # Farben definieren
        colors = []
        if highlight_course:
            for course in top_courses['Kurs']:
                if course == highlight_course:
                    colors.append(COLOR_PALETTE['danger'])  # Hervorgehobener Kurs in Rot
                else:
                    colors.append(COLOR_PALETTE['primary'])  # Andere Kurse in Blau
        
        # Balkendiagramm erstellen
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(
            top_courses['Kurs'], 
            top_courses['Teilnehmerzahl'],
            color=colors if colors else COLOR_PALETTE['primary']
        )
        
        # Beschriftungen und Layout
        ax.set_title('Anzahl der Schüler pro Kurs', fontsize=16)
        ax.set_xlabel('Kurs', fontsize=12)
        ax.set_ylabel('Anzahl der Schüler', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Durchschnittliche Teilnehmerzahl als horizontale Linie
        avg_participants = top_courses['Teilnehmerzahl'].mean()
        ax.axhline(y=avg_participants, color=COLOR_PALETTE['warning'], linestyle='--', 
                   label=f'Durchschnitt: {avg_participants:.1f}')
        
        # Werte über den Balken anzeigen
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=9
            )
        
        # Grid und Legende
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        # Layout optimieren
        plt.tight_layout()
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.perf(f"Balkendiagramm-Erstellung in {end_time - start_time:.2f} Sekunden abgeschlossen")
        
        return fig
    except Exception as e:
        logger.error(f"Fehler bei der Erstellung des Kurs-Balkendiagramms: {e}")
        # Fehlermeldung im Diagramm
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Fehler bei der Diagrammerstellung: {str(e)}", 
                ha='center', va='center', fontsize=12, color='red')
        ax.set_axis_off()
        return fig
        # Gitter und Layout optimieren
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # In BytesIO-Objekt speichern
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Kurs-Schüler-Diagramm: {end_time - start_time:.2f} Sekunden")
        
        return buf
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Kurs-Schüler-Diagramms: {e}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def generate_participant_course_matrix(students_df):
    """Erstellt eine Matrix, die zeigt, welche Schüler in welchen Kursen sind"""
    if students_df.empty:
        return pd.DataFrame()
    
    # Performance-Messung starten
    start_time = time.time()
    
    try:
        # Prüfen, ob Kurse_Liste existiert, andernfalls aus 'Kurse' erstellen
        if 'Kurse_Liste' not in students_df.columns:
            if 'Kurse' in students_df.columns:
                # Kurse von String zu Liste konvertieren
                students_df['Kurse_Liste'] = students_df['Kurse'].apply(
                    lambda x: [k.strip() for k in x.split(',')] if isinstance(x, str) else []
                )
            else:
                logger.error("Weder 'Kurse_Liste' noch 'Kurse' Spalte in den Daten gefunden")
                return pd.DataFrame()
        
        # Alle Kurse aus den Schülerdaten extrahieren
        all_courses = set()
        for courses_list in students_df['Kurse_Liste']:
            # Stelle sicher, dass es eine Liste ist
            if isinstance(courses_list, list):
                all_courses.update(courses_list)
            elif isinstance(courses_list, str):
                # Falls es ein String ist, in Liste umwandeln
                all_courses.update([k.strip() for k in courses_list.split(',')])
        
        # Matrix erstellen: Zeilen = Schüler, Spalten = Kurse
        matrix_data = {course: [] for course in sorted(all_courses)}
        
        for _, student in students_df.iterrows():
            student_courses = student['Kurse_Liste']
            
            # Stelle sicher, dass es eine Liste ist
            if isinstance(student_courses, str):
                student_courses = [k.strip() for k in student_courses.split(',')]
            elif not isinstance(student_courses, list):
                student_courses = []
                
            for course in all_courses:
                matrix_data[course].append(1 if course in student_courses else 0)
        
        # DataFrame erstellen
        df = pd.DataFrame(matrix_data, index=students_df['Name'])
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Matrix-Generierung: {end_time - start_time:.2f} Sekunden")
        
        return df
    
    except Exception as e:
        logger.error(f"Fehler bei der Matrix-Generierung: {e}")
        import traceback
        logger.error(f"Stacktrace: {traceback.format_exc()}")
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

def plot_course_participants_bar(courses_df, num_courses=None, highlight_course=None):
    """Erstellt ein interaktives Balkendiagramm für die Anzahl der Teilnehmer pro Kurs"""
    if courses_df.empty:
        return None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        if num_courses:
            courses_to_plot = courses_df.sort_values('Teilnehmerzahl', ascending=False).head(num_courses)
        else:
            courses_to_plot = courses_df
        
        # Farben anpassen, wenn ein Kurs hervorgehoben werden soll
        if highlight_course:
            colors = [COLOR_PALETTE['primary'] if x != highlight_course else COLOR_PALETTE['danger'] 
                      for x in courses_to_plot['Kurs']]
            color_discrete_map = None
        else:
            colors = None
            color_discrete_map = {
                'Leistungskurs': COLOR_PALETTE['primary'],
                'Grundkurs': COLOR_PALETTE['secondary']
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
            hover_data=['Kurstyp', 'Lehrer'] if 'Lehrer' in courses_to_plot.columns else ['Kurstyp']
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
            margin=dict(l=50, r=50, t=80, b=120),  # Mehr Platz für Kursbezeichnungen
            font=dict(family="Open Sans, sans-serif")
        )
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Balkendiagramm-Generierung: {end_time - start_time:.2f} Sekunden")
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Balkendiagramm-Erstellung: {e}")
        return None

def create_timetable_heatmap(timetable_df):
    """Erstellt eine interaktive Heatmap des Stundenplans"""
    if timetable_df.empty:
        return None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Aggregiere die Anzahl der Fächer pro Tag und Stunde
        heatmap_data = timetable_df.groupby(['Tag', 'Stunde']).size().reset_index(name='Anzahl_Faecher')
        
        # Konvertiere zu einer Pivot-Tabelle für die Heatmap
        pivot_data = heatmap_data.pivot(index='Stunde', columns='Tag', values='Anzahl_Faecher').fillna(0)
        
        # Wochentage in richtige Reihenfolge bringen
        wochentage = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag']
        pivot_data = pivot_data.reindex(columns=wochentage)
        
        # Hover-Text erstellen mit Details zu jedem Zeitslot
        hover_texts = []
        for stunde in pivot_data.index:
            hover_row = []
            for tag in pivot_data.columns:
                count = pivot_data.loc[stunde, tag]
                
                if count > 0:
                    # Alle Kurse für diesen Tag und diese Stunde finden
                    kurse = timetable_df[(timetable_df['Tag'] == tag) & (timetable_df['Stunde'] == stunde)]
                    kurs_liste = kurse['Fach'].tolist()
                    lehrer_liste = kurse['Lehrer'].tolist()
                    raum_liste = kurse['Raum'].tolist()
                    
                    details = "<br>".join([f"{k} ({l}, {r})" for k, l, r in 
                                          zip(kurs_liste, lehrer_liste, raum_liste)])
                    
                    hover_row.append(f"Tag: {tag}<br>Stunde: {stunde}<br>Anzahl Kurse: {int(count)}<br><br>{details}")
                else:
                    hover_row.append(f"Tag: {tag}<br>Stunde: {stunde}<br>Keine Kurse")
            
            hover_texts.append(hover_row)
        
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
        
        # Hover-Text hinzufügen
        fig.update(data=[{
            'hovertext': hover_texts,
            'hoverinfo': 'text'
        }])
        
        fig.update_layout(
            xaxis={'side': 'top'},
            yaxis={'dtick': 1},
            coloraxis_colorbar=dict(
                title="Anzahl Fächer",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
            ),
            height=600,
            margin=dict(l=40, r=40, t=80, b=40),
            font=dict(family="Open Sans, sans-serif"),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Open Sans, sans-serif"
            )
        )
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Stundenplan-Heatmap-Generierung: {end_time - start_time:.2f} Sekunden")
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Stundenplan-Heatmap-Erstellung: {e}")
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

def analyze_teacher_workload(timetable_df, courses_df=None, course_details_df=None):
    """Analysiert die Arbeitsbelastung der Lehrer mit erweiterten Details"""
    if timetable_df.empty:
        return pd.DataFrame()
    
    try:
        # Performance-Messung starten
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
        
        # Wenn Kursdetails und Kursdaten verfügbar sind, diese hinzufügen
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
            
            # Durchschnittliche Schülerzahl pro Kurs
            teacher_avg_students = course_with_teachers.groupby('Lehrer_Kuerzel').agg(
                Durchschnitt_Schüler=('Teilnehmerzahl', 'mean')
            ).reset_index()
            
            # Mit Arbeitsbelastungsdaten zusammenführen
            teacher_workload = pd.merge(
                teacher_workload,
                teacher_students,
                left_on='Lehrer',
                right_on='Lehrer_Kuerzel',
                how='left'
            )
            
            # Durchschnitt hinzufügen
            teacher_workload = pd.merge(
                teacher_workload,
                teacher_avg_students,
                left_on='Lehrer',
                right_on='Lehrer_Kuerzel',
                how='left'
            )
            
            # Fehlende Werte auffüllen
            teacher_workload['Gesamtzahl_Schüler'] = teacher_workload['Gesamtzahl_Schüler'].fillna(0).astype(int)
            teacher_workload['Durchschnitt_Schüler'] = teacher_workload['Durchschnitt_Schüler'].fillna(0).round(1)
            
            # Arbeitslast-Score berechnen (gewichtete Summe aus Stunden, Kursen und Schülern)
            if 'Anzahl_Kurse' in teacher_workload.columns and 'Gesamtzahl_Schüler' in teacher_workload.columns:
                # Normalisierte Werte
                max_stunden = teacher_workload['Anzahl_Stunden'].max() or 1
                max_kurse = teacher_workload['Anzahl_Kurse'].max() or 1
                max_schueler = teacher_workload['Gesamtzahl_Schüler'].max() or 1
                
                teacher_workload['Arbeitslast_Score'] = (
                    (teacher_workload['Anzahl_Stunden'] / max_stunden) * 0.4 +
                    (teacher_workload['Anzahl_Kurse'] / max_kurse) * 0.3 +
                    (teacher_workload['Gesamtzahl_Schüler'] / max_schueler) * 0.3
                ) * 100
                
                teacher_workload['Arbeitslast_Score'] = teacher_workload['Arbeitslast_Score'].round(1)
            
            # Spalten bereinigen
            if 'Lehrer_Kuerzel' in teacher_workload.columns:
                teacher_workload = teacher_workload.drop('Lehrer_Kuerzel', axis=1)
        
        # Nach Arbeitsbelastung sortieren
        teacher_workload = teacher_workload.sort_values('Anzahl_Stunden', ascending=False)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Lehrerarbeitsbelastung-Analyse: {end_time - start_time:.2f} Sekunden")
        
        return teacher_workload
        
    except Exception as e:
        logger.error(f"Fehler bei der Lehrerarbeitsbelastungs-Analyse: {e}")
        return pd.DataFrame()

def get_room_usage(timetable_df):
    """Analysiert die Raumnutzung mit detaillierten Statistiken"""
    if timetable_df.empty:
        return pd.DataFrame()
    
    try:
        # Performance-Messung starten
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
        room_usage['Auslastung_Prozent'] = room_usage['Auslastung_Prozent'].round(1)
        
        # Raumkategorien ermitteln (basierend auf Raumnamen-Präfixen)
        def categorize_room(room):
            room = str(room).lower()
            if room.startswith('pc') or room.startswith('edv') or 'computer' in room:
                return 'Computer-Raum'
            elif 'labor' in room or 'lab' in room:
                return 'Labor'
            elif 'sport' in room or 'turn' in room:
                return 'Sporthalle'
            elif 'musik' in room:
                return 'Musikraum'
            elif 'kunst' in room:
                return 'Kunstraum'
            elif 'biblio' in room:
                return 'Bibliothek'
            elif 'aula' in room:
                return 'Aula'
            else:
                return 'Klassenzimmer'
        
        room_usage['Raumtyp'] = room_usage['Raum'].apply(categorize_room)
        
        # Diverse Fächer pro Raum
        room_subjects = timetable_df.groupby('Raum')['Fach'].nunique().reset_index(name='Verschiedene_Fächer')
        room_usage = pd.merge(room_usage, room_subjects, on='Raum', how='left')
        
        # Nach Nutzungshäufigkeit sortieren
        room_usage = room_usage.sort_values('Nutzungshäufigkeit', ascending=False)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Raumnutzungs-Analyse: {end_time - start_time:.2f} Sekunden")
        
        return room_usage
        
    except Exception as e:
        logger.error(f"Fehler bei der Raumnutzungs-Analyse: {e}")
        return pd.DataFrame()

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
            annotations=annotations,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Open Sans, sans-serif"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Stundenplan-Visualisierung: {e}")
        return None

def search_students_by_name(students_df, search_query):
    """Sucht Schüler anhand ihres Namens"""
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

def search_students_by_courses(students_df, courses):
    """Sucht Schüler anhand ihrer belegten Kurse"""
    if students_df.empty or not courses:
        return pd.DataFrame()
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Liste der Kurse in Kleinbuchstaben umwandeln
        courses_lower = [course.lower() for course in courses]
        
        # Schüler finden, die alle angegebenen Kurse belegen
        matched_students = []
        
        for _, student in students_df.iterrows():
            student_courses_lower = [course.lower() for course in student['Kurse_Liste']]
            
            # Prüfen, ob der Schüler alle angegebenen Kurse belegt
            if all(course in student_courses_lower for course in courses_lower):
                matched_students.append(student)
        
        # DataFrame erstellen
        result_df = pd.DataFrame(matched_students)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Kursbasierte Suche: {end_time - start_time:.2f} Sekunden")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Fehler bei der kursbasierten Suche: {e}")
        return pd.DataFrame()

def calculate_student_overlap(students_df):
    """Berechnet die Überschneidungen zwischen Schülern basierend auf gemeinsamen Kursen.
    
    Args:
        students_df (pd.DataFrame): DataFrame mit Schülerinformationen
        
    Returns:
        pd.DataFrame: Matrix mit Überschneidungen zwischen Schülern
    """
    if students_df is None or students_df.empty:
        logger.warning("Keine Schülerdaten für die Überschneidungsberechnung vorhanden")
        return pd.DataFrame()
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Prüfen, ob erforderliche Spalten vorhanden sind
        if 'Name' not in students_df.columns or 'Kurse_Liste' not in students_df.columns:
            logger.error(f"Erforderliche Spalten fehlen. Vorhanden: {students_df.columns.tolist()}")
            return pd.DataFrame()
        
        # Liste aller Schüler
        students = students_df['Name'].tolist()
        
        # Anzahl der Schüler
        n_students = len(students)
        
        # Vorverarbeitung: Speichere alle Kurslisten in einem Dictionary für schnelleren Zugriff
        student_courses = {}
        for student in students:
            try:
                courses = set(students_df.loc[students_df['Name'] == student, 'Kurse_Liste'].iloc[0])
                student_courses[student] = courses
            except Exception as e:
                logger.warning(f"Fehler beim Verarbeiten der Kursliste für {student}: {e}")
                student_courses[student] = set()
        
        # Matrix erstellen
        overlap_matrix = pd.DataFrame(0, index=students, columns=students)
        
        # Fortschrittsanzeige vorbereiten
        total_pairs = n_students * (n_students + 1) // 2  # Anzahl der Paare inkl. Diagonale
        progress_interval = max(1, total_pairs // 20)  # Alle 5% Fortschritt loggen
        
        # Überschneidungen berechnen
        pair_count = 0
        
        for i, student1 in enumerate(students):
            student1_courses = student_courses[student1]
            
            # Diagonale direkt setzen
            overlap_matrix.loc[student1, student1] = len(student1_courses)
            
            # Nur die obere Dreiecksmatrix berechnen, dann spiegeln
            for j in range(i+1, len(students)):
                student2 = students[j]
                student2_courses = student_courses[student2]
                
                # Gemeinsame Kurse zählen
                overlap = len(student1_courses.intersection(student2_courses))
                
                # In der Matrix speichern (symmetrisch)
                overlap_matrix.loc[student1, student2] = overlap
                overlap_matrix.loc[student2, student1] = overlap
                
                # Fortschritt zählen
                pair_count += 1
                if pair_count % progress_interval == 0:
                    progress_percent = 100 * pair_count / total_pairs
                    logger.perf(f"Überschneidungsberechnung: {progress_percent:.1f}% abgeschlossen")
        
        # Performance-Messung beenden
        end_time = time.time()
        duration = end_time - start_time
        pairs_per_second = total_pairs / duration
        logger.perf(f"Schülerüberschneidungsberechnung: {duration:.2f} Sekunden " +
                  f"({pairs_per_second:.1f} Vergleiche/Sekunde)")
        
        return overlap_matrix
        
    except Exception as e:
        logger.error(f"Fehler bei der Schülerüberschneidungsberechnung: {str(e)}")
        # Stacktrace für bessere Diagnose
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def plot_student_overlap_heatmap(overlap_matrix, title="Kursüberschneidungen zwischen Schülern"):
    """Erstellt eine interaktive Heatmap der Schülerüberschneidungen.
    
    Args:
        overlap_matrix (pd.DataFrame): Matrix mit Überschneidungen zwischen Schülern
        title (str, optional): Titel der Heatmap
        
    Returns:
        plotly.graph_objects.Figure or None: Die erstellte Heatmap oder None bei Fehler
    """
    if overlap_matrix is None or overlap_matrix.empty:
        logger.warning("Keine Überschneidungsdaten zum Visualisieren vorhanden")
        # Leere Figur mit Hinweis erstellen
        fig = go.Figure()
        fig.add_annotation(
            text="Keine Daten verfügbar",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            title="Keine Daten für Kursüberschneidungen verfügbar",
            height=600
        )
        return fig
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Größe basierend auf der Anzahl der Schüler anpassen
        n_students = len(overlap_matrix)
        height = max(600, min(1200, n_students * 15))
        width = max(700, min(1600, n_students * 15))
        
        # Sortieren für bessere Visualisierung
        # Wir sortieren nach der Summe der Überschneidungen, um ähnliche Profile zu gruppieren
        sums = overlap_matrix.sum(axis=1).sort_values(ascending=False)
        sorted_matrix = overlap_matrix.loc[sums.index, sums.index]
        
        fig = px.imshow(
            sorted_matrix,
            labels=dict(x="Schüler", y="Schüler", color="Anzahl gemeinsamer Kurse"),
            x=sorted_matrix.columns,
            y=sorted_matrix.index,
            color_continuous_scale='YlGnBu',
            title=title
        )
        
        fig.update_layout(
            height=height,
            width=width,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis={'side': 'bottom', 'tickangle': -45},
            yaxis={'autorange': 'reversed'},
            coloraxis_colorbar=dict(
                title="Gemeinsame Kurse",
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
        
        # Anpassungen für bessere Anzeige bei vielen Schülern
        if n_students > 30:
            # Zeige nur jeden x-ten Tick auf den Achsen
            tick_interval = max(1, n_students // 20)
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, n_students, tick_interval)),
                    ticktext=[sorted_matrix.columns[i] for i in range(0, n_students, tick_interval)]
                ),
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, n_students, tick_interval)),
                    ticktext=[sorted_matrix.index[i] for i in range(0, n_students, tick_interval)]
                )
            )
        
        # Custom-Hover-Information für bessere Lesbarkeit
        hover_template = "<b>%{y}</b> und <b>%{x}</b><br>" + \
                         "Gemeinsame Kurse: <b>%{z}</b>"
        fig.update_traces(hovertemplate=hover_template)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.perf(f"Schülerheatmap-Erstellung: {end_time - start_time:.2f} Sekunden")
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Schülerheatmap-Erstellung: {str(e)}")
        # Fehlerdiagramm erstellen
        fig = go.Figure()
        fig.add_annotation(
            text=f"Fehler bei der Heatmap-Erstellung:<br>{str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Fehler bei der Visualisierung der Kursüberschneidungen",
            height=600
        )
        return fig
        return None

def generate_student_overlap_svg(overlap_matrix, title="Kursüberschneidungen zwischen Schülern"):
    """Erstellt eine SVG-Datei der Schülerüberschneidungsheatmap mit Matplotlib (nicht-interaktiv)"""
    if overlap_matrix.empty:
        return None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Sicherstellen, dass die Matrix numerische Werte enthält
        # Konvertiere alle Werte zu float, falls sie Objekte oder Strings sein sollten
        numeric_matrix = overlap_matrix.astype(float)
        
        # Erstelle eine eigene Farbpalette für die Heatmap
        colors = ['#F3F4F6', '#DBEAFE', '#93C5FD', '#3B82F6', '#1D4ED8']
        custom_cmap = LinearSegmentedColormap.from_list('custom_YlGnBu', colors)
        
        # Größe der Figur basierend auf der Anzahl der Schüler anpassen
        n_students = len(numeric_matrix)
        figsize = (max(10, n_students/3), max(8, n_students/3))
        
        # Erstelle die Figur und die Heatmap
        plt.figure(figsize=figsize)
        
        # Erstelle die Heatmap
        heatmap = sns.heatmap(
            numeric_matrix,
            cmap=custom_cmap,
            annot=True,  # Werte anzeigen
            fmt="g",     # Generelles Format für Zahlen (unterstützt sowohl Ganzzahlen als auch Dezimalzahlen)
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
        # Erweiterte Fehlerprotokollierung
        import traceback
        logger.error(f"Stacktrace: {traceback.format_exc()}")
        return None

def create_person_network_graph(students_df, focus_person=None, min_shared_courses=2):
    """Erstellt ein Netzwerkdiagramm der Schülerbeziehungen basierend auf gemeinsamen Kursen"""
    if students_df.empty:
        return None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Überschneidungsmatrix berechnen
        overlap_matrix = calculate_student_overlap(students_df)
        
        # Netzwerk erstellen
        G = nx.Graph()
        
        # Knoten (Schüler) hinzufügen
        for student in overlap_matrix.index:
            G.add_node(student, type='student')
        
        # Kanten hinzufügen (nur für Schülerpaare mit mindestens min_shared_courses gemeinsamen Kursen)
        for i, student1 in enumerate(overlap_matrix.index):
            for j, student2 in enumerate(overlap_matrix.columns):
                if i < j:  # Nur die obere Dreiecksmatrix betrachten
                    shared_courses = overlap_matrix.loc[student1, student2]
                    if shared_courses >= min_shared_courses:
                        G.add_edge(student1, student2, weight=shared_courses)
        
        # Wenn ein Fokus-Schüler angegeben ist, nur diesen und direkte Verbindungen anzeigen
        if focus_person and focus_person in G:
            # Nachbarn des Fokus-Schülers finden
            neighbors = list(G.neighbors(focus_person))
            
            # Subgraph mit dem Fokus-Schüler und seinen Nachbarn erstellen
            nodes_to_keep = [focus_person] + neighbors
            G = G.subgraph(nodes_to_keep)
        
        # Layout berechnen (Fruchterman-Reingold für bessere Verteilung)
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Knoten und Kanten für Plotly vorbereiten
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Text und Farbe
            node_text.append(node)
            
            # Größe basierend auf Anzahl der Verbindungen
            size = 10 + len(list(G.neighbors(node)))
            node_size.append(size)
            
            # Farbe: Fokus-Schüler rot, andere blau
            if focus_person and node == focus_person:
                node_color.append(COLOR_PALETTE['danger'])  # Rot für Fokus-Schüler
            else:
                node_color.append(COLOR_PALETTE['primary'])  # Blau für andere Schüler
        
        # Kanten vorbereiten
        edge_x = []
        edge_y = []
        edge_width = []
        edge_hover = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Breite der Kante basierend auf der Anzahl der gemeinsamen Kurse
            weight = G.edges[edge]['weight']
            width = max(1, weight / 2)  # Minimum 1, skaliert nach Gewicht
            edge_width.extend([width, width, None])
            
            # Hover-Text für Kanten
            hover_text = f"{edge[0]} und {edge[1]}: {weight} gemeinsame Kurse"
            edge_hover.extend([hover_text, hover_text, None])
        
        # Kanten zeichnen
        # Erstelle separate Traces für jede Kantenbreite, um verschiedene Linienstärken zu unterstützen
        edge_traces = []
        
        # Gruppiere Kanten nach Breite, um die Anzahl der Traces zu minimieren
        i = 0
        while i < len(edge_x) - 2:  # In Dreiergruppen verarbeiten (x0, x1, None)
            x_vals = [edge_x[i], edge_x[i+1], edge_x[i+2]]
            y_vals = [edge_y[i], edge_y[i+1], edge_y[i+2]]
            width = edge_width[i]  # Nur der erste Wert wird verwendet
            hover = edge_hover[i]  # Nur der erste Wert wird verwendet
            
            edge_trace = go.Scatter(
                x=x_vals, y=y_vals,
                line=dict(width=width, color='rgba(150,150,150,0.7)'),
                hoverinfo='text',
                text=hover,
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
            i += 3
        
        # Knoten zeichnen
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=1, color='rgba(200,200,200,0.8)')
            ),
            text=node_text,
            hoverinfo='text'
        )
        
        # Layout erstellen
        layout = go.Layout(
            title=dict(
                text='Schülernetzwerk basierend auf gemeinsamen Kursen',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='rgba(255,255,255,0.8)',
            paper_bgcolor='white',
            font=dict(family="Open Sans, sans-serif"),
            annotations=[
                dict(
                    text="Größere Knoten: mehr Verbindungen | Dickere Linien: mehr gemeinsame Kurse",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.01, y=-0.01,
                    font=dict(family="Open Sans, sans-serif", size=12)
                )
            ]
        )
        
        # Figure erstellen mit allen Edge-Traces und Node-Trace
        fig = go.Figure(data=[*edge_traces, node_trace], layout=layout)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Schülernetzwerk-Erstellung: {end_time - start_time:.2f} Sekunden")
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Schülernetzwerk-Erstellung: {e}")
        return None

# ----- NEUE FUNKTIONEN -----

def export_plotly_figure(fig, format="png", width=1200, height=800):
    """
    Exportiert eine Plotly-Figur als PNG oder SVG.
    
    Args:
        fig: Plotly Figure-Objekt
        format: Ausgabeformat ('png' oder 'svg')
        width: Breite des Exports in Pixeln
        height: Höhe des Exports in Pixeln
        
    Returns:
        Bytes-Objekt mit der exportierten Figur oder None im Fehlerfall
    """
    if fig is None:
        return None
        
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Format prüfen
        if format.lower() not in ["png", "svg"]:
            logger.warning(f"Unbekanntes Export-Format: {format}. Verwende PNG.")
            format = "png"
            
        # Bild exportieren
        img_bytes = fig.to_image(format=format, width=width, height=height, scale=2)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Plotly-Figure-Export als {format.upper()}: {end_time - start_time:.2f} Sekunden")
        
        return img_bytes
        
    except Exception as e:
        logger.error(f"Fehler beim Exportieren der Plotly-Figur: {e}")
        # Alternativer Export versuchen, falls das Problem mit kaleido zusammenhängt
        try:
            logger.info("Versuche alternativen Export-Weg...")
            if format.lower() == "svg":
                buffer = io.StringIO()
                fig.write_image(buffer, format=format, width=width, height=height, engine="orca")
                return buffer.getvalue().encode('utf-8')
            else:
                buffer = io.BytesIO()
                fig.write_image(buffer, format=format, width=width, height=height, engine="orca")
                buffer.seek(0)
                return buffer.getvalue()
        except Exception as e2:
            logger.error(f"Auch alternativer Export fehlgeschlagen: {e2}")
            return None


def create_student_course_sankey(students_df, max_students=50, min_shared_courses=3):
    """
    Erstellt ein Sankey-Diagramm, das die Schülerströme zwischen Kursen visualisiert.
    
    Args:
        students_df: DataFrame mit Schülerdaten
        max_students: Maximale Anzahl der Schüler für die Analyse
        min_shared_courses: Mindestanzahl gemeinsamer Kurse für die Visualisierung
        
    Returns:
        Plotly Figure-Objekt mit dem Sankey-Diagramm
    """
    if students_df.empty:
        return None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Schüler mit den meisten Kursen auswählen
        students_with_course_count = students_df.copy()
        students_with_course_count['course_count'] = students_with_course_count['Kurse_Liste'].apply(len)
        filtered_students = students_with_course_count.nlargest(max_students, 'course_count')
        
        # Alle Kurse und ihre Häufigkeit zählen
        all_courses = []
        for courses in filtered_students['Kurse_Liste']:
            all_courses.extend(courses)
        
        course_counts = pd.Series(all_courses).value_counts()
        
        # Top-Kurse nach Häufigkeit auswählen
        top_courses = course_counts.nlargest(20).index.tolist()
        
        # Kurspaare finden, die häufig zusammen belegt werden
        course_pairs = []
        
        for _, student in filtered_students.iterrows():
            courses = [course for course in student['Kurse_Liste'] if course in top_courses]
            for i, course1 in enumerate(courses):
                for course2 in courses[i+1:]:
                    course_pairs.append((course1, course2))
        
        # Zählen, wie oft jedes Kurspaar vorkommt
        pair_counts = pd.Series(course_pairs).value_counts()
        
        # Nur Paare mit mindestens min_shared_courses gemeinsamen Schülern verwenden
        significant_pairs = pair_counts[pair_counts >= min_shared_courses]
        
        if significant_pairs.empty:
            logger.warning("Keine signifikanten Kurspaare gefunden.")
            return None
        
        # Knoten und Links für Sankey-Diagramm vorbereiten
        nodes_list = list(set([pair[0] for pair in significant_pairs.index] + 
                              [pair[1] for pair in significant_pairs.index]))
        
        # Node-IDs erstellen
        node_ids = {node: i for i, node in enumerate(nodes_list)}
        
        # Links erstellen
        source = []
        target = []
        value = []
        
        for pair, count in significant_pairs.items():
            source.append(node_ids[pair[0]])
            target.append(node_ids[pair[1]])
            value.append(count)
        
        # Node-Labels mit zusätzlichen Informationen anreichern
        node_labels = []
        for node in nodes_list:
            students_in_course = sum(1 for _, student in filtered_students.iterrows() 
                                    if node in student['Kurse_Liste'])
            node_labels.append(f"{node} ({students_in_course})")
        
        # Farbpalette erstellen (abwechselnd blau und grün)
        node_colors = ['rgba(31, 119, 180, 0.8)' if i % 2 == 0 else 'rgba(44, 160, 44, 0.8)' 
                       for i in range(len(nodes_list))]
        
        # Sankey-Diagramm erstellen
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color='rgba(100, 100, 100, 0.2)'
            )
        )])
        
        # Layout anpassen
        fig.update_layout(
            title_text="Schülerströme zwischen Kursen",
            font=dict(size=12, family="Open Sans, sans-serif"),
            height=800,
            margin=dict(l=25, r=25, t=50, b=25)
        )
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Sankey-Diagramm-Erstellung: {end_time - start_time:.2f} Sekunden")
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der Sankey-Diagramm-Erstellung: {e}")
        return None


def create_3d_student_network(students_df, max_students=100, min_shared_courses=2):
    """
    Erstellt eine 3D-Visualisierung der Schülernetzwerke basierend auf gemeinsamen Kursen.
    
    Args:
        students_df: DataFrame mit Schülerdaten
        max_students: Maximale Anzahl der Schüler für die Analyse
        min_shared_courses: Mindestanzahl gemeinsamer Kurse für eine Verbindung
        
    Returns:
        Plotly Figure-Objekt mit der 3D-Netzwerkvisualisierung
    """
    if students_df.empty:
        return None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Schüler mit den meisten Kursen auswählen
        students_with_course_count = students_df.copy()
        students_with_course_count['course_count'] = students_with_course_count['Kurse_Liste'].apply(len)
        filtered_students = students_with_course_count.nlargest(max_students, 'course_count')
        
        # Überschneidungsmatrix berechnen
        overlap_matrix = calculate_student_overlap(filtered_students)
        
        # Diagonale (Selbstüberschneidung) auf 0 setzen
        np.fill_diagonal(overlap_matrix.values, 0)
        
        # Graph erstellen
        G = nx.Graph()
        
        # Knoten (Schüler) hinzufügen
        for student in overlap_matrix.index:
            G.add_node(student, type='student')
        
        # Kanten hinzufügen (nur für Schülerpaare mit mindestens min_shared_courses gemeinsamen Kursen)
        for i, student1 in enumerate(overlap_matrix.index):
            for j, student2 in enumerate(overlap_matrix.columns):
                if i < j:  # Nur die obere Dreiecksmatrix betrachten
                    shared_courses = overlap_matrix.loc[student1, student2]
                    if shared_courses >= min_shared_courses:
                        G.add_edge(student1, student2, weight=shared_courses)
        
        # Wenn es keine Kanten gibt, frühzeitig beenden
        if len(G.edges()) == 0:
            logger.warning("Keine ausreichenden Verbindungen für ein 3D-Netzwerk gefunden.")
            return None
        
        # 3D-Layout mit NetworkX und Fruchterman-Reingold
        pos_3d = nx.spring_layout(G, dim=3, seed=42)
        
        # Knoten für Plotly vorbereiten
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y, z = pos_3d[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            # Text und Größe
            course_count = len(filtered_students.loc[filtered_students['Name'] == node, 'Kurse_Liste'].iloc[0])
            connection_count = len(list(G.neighbors(node)))
            node_text.append(f"{node}<br>Kurse: {course_count}<br>Verbindungen: {connection_count}")
            
            # Größe basierend auf Verbindungen
            node_size.append(5 + connection_count * 2)
            
            # Farbe basierend auf Kursanzahl
            node_color.append(course_count)
        
        # Kanten für Plotly vorbereiten
        edge_x = []
        edge_y = []
        edge_z = []
        edge_text = []
        
        for edge in G.edges():
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            
            # Linien zeichnen
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
            # Kantengewicht für Hover-Text
            weight = G.edges[edge]['weight']
            edge_text.extend([f"{edge[0]} - {edge[1]}: {weight} gemeinsame Kurse"] * 3)
        
        # Kanten-Trace erstellen
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(120, 120, 120, 0.2)', width=1),
            text=edge_text,
            hoverinfo='text'
        )
        
        # Knoten-Trace erstellen
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title="Kursanzahl",
                    thickness=15,
                    len=0.5
                )
            ),
            text=node_text,
            hoverinfo='text'
        )
        
        # Figure erstellen
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Layout anpassen
        fig.update_layout(
            title="3D-Visualisierung des Schülernetzwerks",
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=800,
            showlegend=False,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Open Sans, sans-serif"
            )
        )
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"3D-Netzwerk-Erstellung: {end_time - start_time:.2f} Sekunden")
        
        return fig
        
    except Exception as e:
        logger.error(f"Fehler bei der 3D-Netzwerk-Erstellung: {e}")
        return None


def perform_student_clustering(students_df, n_clusters=5, algorithm='kmeans'):
    """
    Führt eine Clusteranalyse für Schülergruppen basierend auf Kursüberschneidungen durch.
    
    Args:
        students_df: DataFrame mit Schülerdaten
        n_clusters: Anzahl der zu bildenden Cluster
        algorithm: Clusteralgorithmus ('kmeans', 'hierarchical', 'dbscan')
        
    Returns:
        DataFrame mit Schülerdaten und Cluster-Labels, Figure-Objekt mit Clustervisualisierung
    """
    if students_df.empty:
        return None, None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Überschneidungsmatrix berechnen
        overlap_matrix = calculate_student_overlap(students_df)
        
        # Diagonale (Selbstüberschneidung) auf 0 setzen
        np.fill_diagonal(overlap_matrix.values, 0)
        
        # Distanzmatrix erstellen (1 - normalisierte Ähnlichkeit)
        max_overlap = overlap_matrix.values.max()
        if max_overlap > 0:
            distance_matrix = 1 - (overlap_matrix.values / max_overlap)
        else:
            # Wenn keine Überschneidungen vorhanden sind
            distance_matrix = np.ones_like(overlap_matrix.values)
        
        # Dimensionsreduktion mit MDS für Visualisierung
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distance_matrix)
        
        # Clustering durchführen
        if algorithm.lower() == 'kmeans':
            # Für K-Means brauchen wir Feature-Vektoren
            # Wir verwenden die MDS-Positionen als Features
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(positions)
        
        elif algorithm.lower() == 'hierarchical':
            # Hierarchisches Clustering mit der Distanzmatrix
            clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
            cluster_labels = clustering.fit_predict(distance_matrix)
        
        elif algorithm.lower() == 'dbscan':
            # DBSCAN mit MDS-Positionen
            # Epsilon bestimmen basierend auf Daten
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(positions)
            distances, _ = nn.kneighbors(positions)
            epsilon = np.percentile(distances[:, 1], 90)  # 90. Perzentil als Epsilon
            
            clustering = DBSCAN(eps=epsilon, min_samples=3)
            cluster_labels = clustering.fit_predict(positions)
            
            # -1 sind Ausreißer, wir setzen sie auf den höchsten Cluster-Index + 1
            if -1 in cluster_labels:
                highest_label = max(cluster_labels)
                cluster_labels[cluster_labels == -1] = highest_label + 1
                
            # Tatsächliche Anzahl der Cluster aktualisieren
            n_clusters = len(set(cluster_labels))
        
        else:
            raise ValueError(f"Unbekannter Algorithmus: {algorithm}")
        
        # Ergebnisse in DataFrame speichern
        result_df = students_df.copy()
        result_df['Cluster'] = cluster_labels
        result_df['x_position'] = positions[:, 0]
        result_df['y_position'] = positions[:, 1]
        
        # Visualisierung der Cluster
        # Farbskala für Cluster erstellen
        colors = px.colors.qualitative.Plotly
        cluster_colors = [colors[i % len(colors)] for i in range(n_clusters)]
        
        # Clustergrößen berechnen
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        
        # Figure erstellen
        fig = go.Figure()
        
        # Jeden Cluster als separate Scatter-Trace hinzufügen
        for i in range(n_clusters):
            cluster_data = result_df[result_df['Cluster'] == i]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['x_position'],
                y=cluster_data['y_position'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=cluster_colors[i],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=cluster_data['Name'],
                name=f'Cluster {i} ({len(cluster_data)} Schüler)',
                hovertemplate='%{text}<br>Cluster: %{legendgroup}<extra></extra>'
            ))
        
        # Layout anpassen
        fig.update_layout(
            title=f"Schüler-Clustering ({n_clusters} Cluster, {algorithm})",
            xaxis=dict(title='Dimension 1', showgrid=True, zeroline=True),
            yaxis=dict(title='Dimension 2', showgrid=True, zeroline=True),
            height=700,
            hovermode='closest',
            legend=dict(
                title='Cluster',
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Clustering ({algorithm}): {end_time - start_time:.2f} Sekunden")
        
        return result_df, fig
        
    except Exception as e:
        logger.error(f"Fehler beim Clustering: {e}")
        return None, None


def generate_pdf_report(students_df, courses_df, overlap_matrix=None, charts=None, title="Kursplan-Analyse Bericht"):
    """
    Generiert einen professionellen PDF-Bericht mit den wichtigsten Analyse-Ergebnissen.
    
    Args:
        students_df: DataFrame mit Schülerdaten
        courses_df: DataFrame mit Kursdaten
        overlap_matrix: Matrix mit Überschneidungen (optional)
        charts: Liste mit bereits erstellten Diagrammen als PNG/JPG-Bytes (optional)
        title: Titel des Berichts
        
    Returns:
        PDF-Datei als Bytes-Objekt
    """
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Buffer für PDF-Dokument
        buffer = BytesIO()
        
        # PDF-Dokument erstellen (A4-Format mit angepassten Rändern)
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4,
            title=title,
            author="Kursplan-Analyse Tool",
            creator="Kursplan-Analyse Tool",
            rightMargin=50, leftMargin=50,  # Schmalere Ränder für mehr Platz
            topMargin=60, bottomMargin=50
        )
        
        # Erweiterte Stile für das Dokument
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Title', 
            parent=styles['Title'], 
            fontSize=24, 
            spaceAfter=12,
            textColor=colors.HexColor('#2563EB'),  # Primärfarbe aus COLOR_PALETTE
            alignment=1  # Zentriert
        ))
        styles.add(ParagraphStyle(
            name='Heading1', 
            parent=styles['Heading1'], 
            fontSize=18, 
            spaceAfter=12,
            textColor=colors.HexColor('#2563EB'),
            spaceBefore=15,
            borderWidth=0,
            borderColor=colors.HexColor('#2563EB'),
            borderPadding=(0, 0, 2, 0),  # Unterstreichung
            borderRadius=None
        ))
        styles.add(ParagraphStyle(
            name='Heading2', 
            parent=styles['Heading2'], 
            fontSize=14, 
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#1F2937')  # Dunkler Text für besseren Kontrast
        ))
        styles.add(ParagraphStyle(
            name='BodyText', 
            parent=styles['BodyText'], 
            fontSize=10, 
            spaceAfter=6,
            leading=14  # Größerer Zeilenabstand für bessere Lesbarkeit
        ))
        styles.add(ParagraphStyle(
            name='Caption', 
            parent=styles['BodyText'], 
            fontSize=9, 
            spaceAfter=12,
            leading=12,
            alignment=1,  # Zentriert
            textColor=colors.HexColor('#6B7280')  # Gedämpfte Textfarbe
        ))
        
        # Elemente für das Dokument
        elements = []
        
        # Titel und Header mit moderner Gestaltung
        elements.append(Paragraph(title, styles['Title']))
        elements.append(Paragraph(f"Erstellt am: {datetime.now().strftime('%d.%m.%Y, %H:%M')} Uhr", styles['BodyText']))
        elements.append(Spacer(1, 0.8*cm))
        
        # Zusammenfassung mit modernem Design
        elements.append(Paragraph("Zusammenfassung", styles['Heading1']))
        
        # Erweiterte Zusammenfassungsdaten mit mehr Informationen
        summary_data = [
            ["Anzahl Schüler:", str(len(students_df))],
            ["Anzahl Kurse:", str(len(courses_df)) if not courses_df.empty else "N/A"],
            ["Durchschnittliche Kurse pro Schüler:", f"{students_df['Kurse_Liste'].apply(len).mean():.1f}"],
            ["Max. Kurse pro Schüler:", f"{students_df['Kurse_Liste'].apply(len).max()}"],
            ["Min. Kurse pro Schüler:", f"{students_df['Kurse_Liste'].apply(len).min()}"],
            ["Erstellungsdatum:", datetime.now().strftime("%d.%m.%Y")],
            ["Erstellt mit:", "Kursplan-Analyse Tool v2.0"]
        ]
        
        # Moderne Tabelle mit besserer Formatierung
        summary_table = Table(summary_data, colWidths=[220, 240])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),  # Hellerer Hintergrund
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1F2937')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Mehr Abstand für bessere Lesbarkeit
            ('TOPPADDING', (0, 0), (-1, -1), 8),  # Mehr Abstand für bessere Lesbarkeit
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),  # Feinere Rasterlinien
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Vertikale Zentrierung
            ('ROUNDEDCORNERS', [5, 5, 5, 5]),  # Abgerundete Ecken (falls unterstützt)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.8*cm))
        
        # Schüler mit den meisten Kursen - modernes Design
        elements.append(Paragraph("Top-Schüler nach Kursanzahl", styles['Heading2']))
        
        # Schüler-Kurs-Anzahl berechnen
        students_with_course_count = students_df.copy()
        students_with_course_count['Kursanzahl'] = students_with_course_count['Kurse_Liste'].apply(len)
        top_students = students_with_course_count.nlargest(10, 'Kursanzahl')[['Name', 'Kursanzahl']]
        
        # Verbesserte Tabellenformatierung
        top_students_data = [["Schüler", "Anzahl Kurse"]]
        for _, row in top_students.iterrows():
            top_students_data.append([row['Name'], str(row['Kursanzahl'])])
        
        top_students_table = Table(top_students_data, colWidths=[350, 100])
        top_students_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),  # Farbiger Header
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Weiße Schrift im Header
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Mehr Platz
            ('TOPPADDING', (0, 0), (-1, -1), 8),     # Mehr Platz
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),  # Weißer Hintergrund für Datenzeilen
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),  # Feinere Rasterlinien
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),  # Abwechselnde Zeilenhintergründe
        ]))
        
        elements.append(top_students_table)
        elements.append(Spacer(1, 0.8*cm))
        
        # Wenn Charts vorhanden sind, diese mit verbessertem Layout hinzufügen
        if charts and len(charts) > 0:
            elements.append(Paragraph("Visualisierungen", styles['Heading1']))
            
            for i, chart_data in enumerate(charts):
                try:
                    # Seitenumbruch für besseres Layout bei mehreren Diagrammen
                    if i > 0:
                        elements.append(PageBreak())
                    
                    # Diagramm-Titel basierend auf Index
                    chart_titles = [
                        "Schülerüberschneidungen (Heatmap)",
                        "Schülerströme zwischen Kursen (Sankey)",
                        "Schülernetzwerk basierend auf gemeinsamen Kursen",
                        "Cluster-Analyse der Schülerkursüberschneidungen"
                    ]
                    
                    chart_title = chart_titles[i] if i < len(chart_titles) else f"Visualisierung {i+1}"
                    elements.append(Paragraph(chart_title, styles['Heading2']))
                    elements.append(Spacer(1, 0.3*cm))
                    
                    # Chart-Daten in ein ReportLab-Image umwandeln mit verbesserter Größe
                    img = RLImage(BytesIO(chart_data), width=480, height=320)  # Größere Darstellung
                    elements.append(img)
                    elements.append(Spacer(1, 0.3*cm))
                    elements.append(Paragraph(f"Abbildung {i+1}: {chart_title}", styles['Caption']))
                    elements.append(Spacer(1, 0.5*cm))
                    
                    # Kurze Beschreibung der Visualisierung hinzufügen
                    chart_descriptions = [
                        "Diese Heatmap zeigt die Überschneidungen zwischen Schülern basierend auf gemeinsamen Kursen. "
                        "Je dunkler ein Feld, desto mehr Kurse teilen sich die entsprechenden Schüler.",
                        
                        "Das Sankey-Diagramm visualisiert die Schülerströme zwischen verschiedenen Kursen. "
                        "Die Breite der Verbindungen entspricht der Anzahl der Schüler, die beide Kurse belegen.",
                        
                        "Das Netzwerkdiagramm stellt Verbindungen zwischen Schülern dar, die gemeinsame Kurse belegen. "
                        "Größere Knoten zeigen Schüler mit mehr Verbindungen, dickere Linien mehr gemeinsame Kurse.",
                        
                        "Die Cluster-Analyse gruppiert Schüler basierend auf ihren Kursüberschneidungen. "
                        "Schüler im gleichen Cluster haben ähnliche Kursbelegungsmuster."
                    ]
                    
                    if i < len(chart_descriptions):
                        elements.append(Paragraph(chart_descriptions[i], styles['BodyText']))
                        elements.append(Spacer(1, 0.5*cm))
                    
                except Exception as img_error:
                    logger.error(f"Fehler beim Hinzufügen des Charts {i+1}: {img_error}")
                    elements.append(Paragraph(f"Visualisierung {i+1} konnte nicht geladen werden.", styles['BodyText']))
        
        # Wenn Überschneidungsmatrix vorhanden ist, Top-Überschneidungen mit verbessertem Design hinzufügen
        if overlap_matrix is not None and not overlap_matrix.empty:
            elements.append(Paragraph("Top-Kursüberschneidungen", styles['Heading2']))
            
            # Diagonale (Selbstüberschneidung) auf 0 setzen
            np.fill_diagonal(overlap_matrix.values, 0)
            
            # Top-Überschneidungen finden
            overlap_pairs = []
            
            for i, row in enumerate(overlap_matrix.index):
                for j, col in enumerate(overlap_matrix.columns):
                    if i < j:  # Um Duplikate zu vermeiden
                        overlap_pairs.append({
                            'Kurs1': row,
                            'Kurs2': col,
                            'Überschneidung': overlap_matrix.iloc[i, j]
                        })
            
            overlap_df = pd.DataFrame(overlap_pairs)
            
            # Tabellendaten vorbereiten
            if 'Überschneidung' in overlap_df.columns and not overlap_df.empty:
                top_overlaps = overlap_df.sort_values('Überschneidung', ascending=False).head(15)
                
                if not top_overlaps.empty:
                    overlap_data = [["Kurs 1", "Kurs 2", "Gemeinsame Schüler"]]
                    
                    for _, row in top_overlaps.iterrows():
                        if row['Überschneidung'] > 0:
                            overlap_data.append([
                                row['Kurs1'], 
                                row['Kurs2'], 
                                str(int(row['Überschneidung']))
                            ])
                    
                    overlap_table = Table(overlap_data, colWidths=[180, 180, 100])
                    overlap_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),  # Blauer Header
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Weiße Schrift im Header
                        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
                        ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),  # Mehr Platz
                        ('TOPPADDING', (0, 0), (-1, -1), 8),     # Mehr Platz
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),  # Weißer Hintergrund für Datenzeilen
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),  # Feinere Rasterlinien
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),  # Abwechselnde Zeilenhintergründe
                    ]))
                    
                    elements.append(overlap_table)
                    elements.append(Spacer(1, 0.5*cm))
        
        # Verbesserte Datenschutzhinweise mit Icon-Effekt
        elements.append(PageBreak())
        elements.append(Paragraph("Datenschutz und Informationen", styles['Heading1']))
        elements.append(Spacer(1, 0.5*cm))
        
        # Datenschutzhinweis mit modernem Design
        privacy_data = [
            ["⚠️ Datenschutzhinweis", 
             "Dieser Bericht enthält personenbezogene Daten und unterliegt den Bestimmungen der "
             "DSGVO. Eine Weitergabe ist nur an autorisierte Personen gestattet. "
             "Alle Daten wurden anonymisiert, soweit dies für die Analyse möglich war."]
        ]
        
        privacy_table = Table(privacy_data, colWidths=[120, 430])
        privacy_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#FEF2F2')),  # Leichtes Rot für Warnung
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#FEFEFE')),  # Weißer Hintergrund
            ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor('#B91C1C')),   # Rote Schrift für Warnung
            ('TEXTCOLOR', (1, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (0, 0), 'CENTER'),
            ('VALIGN', (0, 0), (1, 0), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica'),
            ('FONTSIZE', (0, 0), (0, 0), 12),
            ('FONTSIZE', (1, 0), (1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('TOPPADDING', (0, 0), (1, 0), 12),
            ('GRID', (0, 0), (1, 0), 0.5, colors.HexColor('#E5E7EB')),
        ]))
        
        elements.append(privacy_table)
        elements.append(Spacer(1, 0.8*cm))
        
        # Zusätzliche Informationen
        info_data = [
            ["ℹ️ Allgemeine Informationen", 
             "Dieser Bericht wurde automatisch generiert und dient der Analyse von Kursüberschneidungen "
             "und Schülernetzwerken. Die Daten können für die Kurs- und Raumplanung sowie für "
             "pädagogische Analysen verwendet werden."],
            ["📊 Datengrundlage", 
             f"Der Bericht basiert auf {len(students_df) if students_df is not None else 'N/A'} Schülern "
             f"und {len(courses_df) if courses_df is not None else 'N/A'} Kursen."],
            ["📅 Erstellungsdatum", 
             f"{datetime.now().strftime('%d.%m.%Y um %H:%M Uhr')}"]
        ]
        
        info_table = Table(info_data, colWidths=[120, 430])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F0F9FF')),  # Leichtes Blau für Info
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#FEFEFE')),  # Weißer Hintergrund
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#0369A1')),   # Blaue Schrift für Info
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('VALIGN', (0, 0), (1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (0, -1), 12),
            ('FONTSIZE', (1, 0), (1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (1, -1), 10),
            ('TOPPADDING', (0, 0), (1, -1), 10),
            ('GRID', (0, 0), (1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ]))
        
        elements.append(info_table)
        
        # Erweiterte Fußzeile mit mehr Informationen
        elements.append(Spacer(1, 1*cm))
        current_date = datetime.now().strftime("%d.%m.%Y")
        elements.append(Paragraph(
            f"Kursplan-Analyse Tool • Version 1.0 • Generiert am {current_date}",
            ParagraphStyle(
                name='Footer',
                parent=styles['Normal'],
                textColor=colors.HexColor('#6B7280'),
                fontSize=8,
                alignment=1  # Zentriert
            )
        ))
        
        # PDF-Dokument erstellen
        doc.build(elements)
        
        # Buffer zurückspulen und Inhalt auslesen
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"PDF-Bericht-Generierung: {end_time - start_time:.2f} Sekunden")
        
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"Fehler bei der PDF-Generierung: {e}")
        return None


def generate_qr_code(data, name="qrcode", size=200, logo_path=None, fill_color="black", back_color="white"):
    """
    Generiert einen QR-Code mit verbesserten Designoptionen und optionalem Logo.
    
    Args:
        data: Zu kodierende Daten (String oder Dict)
        name: Name für den QR-Code (für Anzeige)
        size: Größe des QR-Codes in Pixeln
        logo_path: Optionaler Pfad zu einem Logo, das in der Mitte des QR-Codes angezeigt wird
        fill_color: Farbe des QR-Codes (als Farbname oder Hex-Code)
        back_color: Hintergrundfarbe des QR-Codes (als Farbname oder Hex-Code)
        
    Returns:
        BytesIO-Objekt mit dem QR-Code als PNG-Bild
    """
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Daten vorbereiten
        if data is None:
            data = "https://example.com"
            logger.warning("Keine Daten für QR-Code angegeben, verwende Standarddaten")
        
        # Wenn data ein Dictionary ist, in JSON konvertieren
        if isinstance(data, dict):
            try:
                import json
                data = json.dumps(data, ensure_ascii=False)
            except Exception as json_error:
                logger.error(f"Fehler bei der JSON-Konvertierung: {json_error}")
                data = str(data)  # Fallback zur String-Konvertierung
        
        # QR-Code mit höherer Fehlerkorrektur erstellen (für bessere Lesbarkeit mit Logo)
        qr = qrcode.QRCode(
            version=1,  # Anfangsgröße, wird automatisch angepasst
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # Höchste Fehlerkorrektur
            box_size=10,
            border=4,
        )
        
        # Daten zum QR-Code hinzufügen
        qr.add_data(data)
        qr.make(fit=True)
        
        # QR-Code in Bild umwandeln mit angegebenen Farben
        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        
        # Umwandlung in RGBA für Transparenz-Unterstützung
        qr_img = img.convert('RGBA')
        
        # Logo hinzufügen, falls angegeben
        if logo_path:
            try:
                if not os.path.exists(logo_path):
                    logger.warning(f"Logo-Datei nicht gefunden: {logo_path}")
                else:
                    # Logo laden und in RGBA umwandeln für Transparenz
                    logo = Image.open(logo_path).convert('RGBA')
                    
                    # QR-Code-Größe ermitteln
                    qr_width, qr_height = qr_img.size
                    
                    # Logo auf maximal 20% der QR-Code-Größe skalieren
                    logo_max_size = int(min(qr_width, qr_height) * 0.2)
                    logo_width, logo_height = logo.size
                    
                    # Seitenverhältnis beibehalten
                    if logo_width > logo_height:
                        new_width = logo_max_size
                        new_height = int(logo_height * (logo_max_size / logo_width))
                    else:
                        new_height = logo_max_size
                        new_width = int(logo_width * (logo_max_size / logo_height))
                    
                    # Logo resizen mit hoher Qualität
                    logo = logo.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Weißer Hintergrund für das Logo erstellen (etwas größer als das Logo)
                    padding = 5
                    bg_size = (new_width + padding*2, new_height + padding*2)
                    logo_bg = Image.new('RGBA', bg_size, (255, 255, 255, 230))
                    
                    # Positionen berechnen (zentriert)
                    bg_pos = (
                        (qr_width - bg_size[0]) // 2,
                        (qr_height - bg_size[1]) // 2
                    )
                    logo_pos = (
                        (qr_width - new_width) // 2,
                        (qr_height - new_height) // 2
                    )
                    
                    # Hintergrund und Logo auf QR-Code zeichnen
                    qr_img.paste(logo_bg, bg_pos, logo_bg)
                    qr_img.paste(logo, logo_pos, logo)
            except Exception as logo_error:
                logger.error(f"Fehler beim Hinzufügen des Logos: {logo_error}")
                # Fortfahren ohne Logo
        
        # Größe anpassen, falls gewünscht
        if size != 200 and size > 0:
            current_size = qr_img.size[0]
            if current_size != size:
                # Gleichmäßige Skalierung berechnen
                scale_factor = size / current_size
                new_size = (
                    int(qr_img.size[0] * scale_factor), 
                    int(qr_img.size[1] * scale_factor)
                )
                # Mit hoher Qualität skalieren
                qr_img = qr_img.resize(new_size, Image.LANCZOS)
        
        # In BytesIO speichern
        buffer = BytesIO()
        qr_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"QR-Code-Generierung erfolgreich: {end_time - start_time:.2f} Sekunden")
        
        return buffer
    
    except Exception as e:
        # Ausführlicher Fehler für bessere Diagnose
        logger.error(f"Fehler bei der QR-Code-Generierung: {e}")
        
        # Fallback: Einfachen QR-Code ohne Extras generieren
        try:
            logger.info("Versuche Fallback-QR-Code zu generieren...")
            
            # Einfachster QR-Code mit Standardeinstellungen
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            
            # Sicherstellen, dass die Daten ein String sind
            qr.add_data(str(data))
            qr.make(fit=True)
            
            # Einfaches Schwarz-Weiß-Bild
            img = qr.make_image(fill_color="black", back_color="white")
            
            # In BytesIO speichern
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            
            logger.info("Fallback-QR-Code erfolgreich generiert")
            return buffer
            
        except Exception as fallback_error:
            logger.error(f"Auch Fallback-QR-Code fehlgeschlagen: {fallback_error}")
            return None


def anonymize_student_data(students_df, anonymization_level='medium'):
    """
    Anonymisiert Schülerdaten gemäß DSGVO-Anforderungen.
    
    Args:
        students_df: DataFrame mit Schülerdaten
        anonymization_level: Grad der Anonymisierung ('low', 'medium', 'high')
        
    Returns:
        DataFrame mit anonymisierten Daten und Info-Dict
    """
    if students_df.empty:
        return None, None
    
    try:
        # Performance-Messung starten
        start_time = time.time()
        
        # Kopie des DataFrames erstellen
        anon_df = students_df.copy()
        
        # Für Nachverfolgung der Anonymisierung
        anon_info = {
            'original_count': len(students_df),
            'anonymized_count': len(anon_df),
            'anonymization_level': anonymization_level,
            'timestamp': datetime.now().isoformat()
        }
        
        # Anonymisierung basierend auf Level
        if anonymization_level == 'low':
            # Nur Nachnamen teilweise anonymisieren
            anon_df['Name'] = anon_df['Name'].apply(
                lambda name: ' '.join([
                    part if i == 0 else part[0] + '*' * (len(part) - 1)
                    for i, part in enumerate(name.split())
                ])
            )
        elif anonymization_level == 'medium':
            # Nur Initialen behalten
            anon_df['Name'] = anon_df['Name'].apply(
                lambda name: '.'.join([part[0] for part in name.split()])
            )
        elif anonymization_level == 'high':
            # Vollständige Anonymisierung mit eindeutigen IDs
            anon_df['Name'] = [f"Student_{uuid.uuid4().hex[:8]}" for _ in range(len(anon_df))]
        
        # Performance-Messung beenden
        end_time = time.time()
        logger.info(f"Daten-Anonymisierung ({anonymization_level}): {end_time - start_time:.2f} Sekunden")
        
        return anon_df, anon_info
        
    except Exception as e:
        logger.error(f"Fehler bei der Daten-Anonymisierung: {e}")
        return None, None


# Audit-Logger für DSGVO-Compliance
class AuditLogger:
    """
    Verwaltet Audit-Logs für DSGVO-Compliance.
    Protokolliert alle Datenoperationen mit Zeitstempel und Details.
    """
    
    def __init__(self, log_file=None):
        self.log_file = log_file or os.path.join(os.path.dirname(__file__), 'audit_log.json')
        
        # In-Memory-Logs für Demo ohne Dateizugriff
        self.logs = []
        
        # Beispielhafte Logs für die Demo
        self.add_demo_logs()
    
    def add_demo_logs(self):
        """Fügt Beispiel-Logs für die Demo hinzu"""
        self.logs = [
            {
                "timestamp": (datetime.now() - pd.Timedelta(hours=2)).isoformat(),
                "action": "view",
                "data_type": "student",
                "user": "admin",
                "details": {"page": "Schülersuche", "student_count": 120}
            },
            {
                "timestamp": (datetime.now() - pd.Timedelta(hours=1)).isoformat(),
                "action": "export",
                "data_type": "course",
                "user": "admin",
                "details": {"format": "csv", "course_count": 45}
            },
            {
                "timestamp": datetime.now().isoformat(),
                "action": "anonymize",
                "data_type": "student",
                "user": "admin",
                "details": {"level": "medium", "student_count": 120}
            },
            {
                "timestamp": (datetime.now() - pd.Timedelta(days=1)).isoformat(),
                "action": "session_end",
                "data_type": "session",
                "user": "anonymous",
                "details": {"duration_minutes": 45}
            }
        ]
    
    def log_action(self, action, data_type, details=None, user="anonymous"):
        """
        Protokolliert eine Aktion im Audit-Log
        
        Args:
            action: Art der Aktion (view, export, anonymize, etc.)
            data_type: Art der Daten (student, course, etc.)
            details: Zusätzliche Details (dict)
            user: Benutzer, der die Aktion ausführt
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data_type": data_type,
            "user": user,
            "details": details or {}
        }
        
        # Log in die in-Memory-Liste einfügen
        self.logs.append(log_entry)
        
        # Wenn eine Datei angegeben ist, speichern
        if self.log_file:
            try:
                with open(self.log_file, 'w') as f:
                    json.dump(self.logs, f, indent=2)
            except Exception as e:
                logger.error(f"Fehler beim Speichern des Audit-Logs: {e}")
        
        logger.info(f"Audit-Log: {action} auf {data_type} durch {user}")
    
    def get_logs(self, filter_action=None, filter_data_type=None, filter_user=None):
        """
        Gibt gefilterte Logs zurück
        
        Args:
            filter_action: Optional Filter für Aktionen
            filter_data_type: Optional Filter für Datentypen
            filter_user: Optional Filter für Benutzer
            
        Returns:
            Liste mit gefilterten Log-Einträgen
        """
        filtered_logs = self.logs
        
        if filter_action:
            filtered_logs = [log for log in filtered_logs if log["action"] == filter_action]
        
        if filter_data_type:
            filtered_logs = [log for log in filtered_logs if log["data_type"] == filter_data_type]
        
        if filter_user:
            filtered_logs = [log for log in filtered_logs if log["user"] == filter_user]
        
        # Nach Zeitstempel sortieren (neueste zuerst)
        return sorted(filtered_logs, key=lambda x: x["timestamp"], reverse=True)
        """
        Gibt gefilterte Logs zurück
        
        Args:
            filter_action: Nach Aktion filtern (optional)
            filter_data_type: Nach Datentyp filtern (optional)
            filter_user: Nach Benutzer filtern (optional)
            limit: Maximale Anzahl zurückgegebener Logs
            
        Returns:
            Liste der gefilterten Logs
        """
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    all_logs = json.load(f)
                
                # Logs filtern
                filtered_logs = all_logs
                
                if filter_action:
                    filtered_logs = [log for log in filtered_logs if log.get("action") == filter_action]
                
                if filter_data_type:
                    filtered_logs = [log for log in filtered_logs if log.get("data_type") == filter_data_type]
                
                if filter_user:
                    filtered_logs = [log for log in filtered_logs if log.get("user") == filter_user]
                
                # Nach Datum sortieren (neueste zuerst)
                filtered_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                # Auf Limit beschränken
                return filtered_logs[:limit]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Audit-Logs: {e}")
            return []
    
    def close(self):
        """Schließt die Audit-Log-Sitzung"""
        try:
            session_end = datetime.now().isoformat()
            session_duration = (datetime.fromisoformat(session_end) - 
                               datetime.fromisoformat(self.session_start)).total_seconds()
            
            # Sitzungsende protokollieren
            self.log_action(
                action="session_end",
                data_type="session",
                details={
                    "session_start": self.session_start,
                    "session_end": session_end,
                    "duration_seconds": session_duration
                }
            )
            
            # Sicherstellen, dass alle Logs gespeichert sind
            self._save_logs()
            
        except Exception as e:
            logger.error(f"Fehler beim Schließen der Audit-Log-Sitzung: {e}")
