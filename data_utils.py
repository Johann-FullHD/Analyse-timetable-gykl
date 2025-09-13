import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Logging-Konfiguration
logger = logging.getLogger(__name__)

# Cache-Gültigkeit in Sekunden
CACHE_TTL = 3600

class UserProfileManager:
    """Klasse zur Verwaltung von Benutzerprofilen und Berechtigungen.
    
    Diese Klasse bietet Funktionen zum Laden, Speichern und Verwalten von Benutzerprofilen
    sowie zur Authentifizierung und Autorisierung von Benutzern.
    
    Attributes:
        storage_path (str): Pfad zur JSON-Datei, in der die Profile gespeichert werden
        profiles (dict): Dictionary mit Benutzerprofilen
        active_user (str): Aktuell aktiver Benutzer
    """
    
    def __init__(self, storage_path: str = "profiles.json") -> None:
        """Initialisiert den UserProfileManager.
        
        Args:
            storage_path: Pfad zur JSON-Datei für die Benutzerprofile
        """
        self.storage_path = storage_path
        self.profiles = {}
        self.active_user = None
        self.load_profiles()
    
    def load_profiles(self) -> None:
        """Lädt Benutzerprofile aus der Speicherdatei.
        
        Falls die Datei nicht existiert oder fehlerhaft ist, werden Standardprofile erstellt.
        """
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.profiles = json.load(f)
                logger.info(f"{len(self.profiles)} Benutzerprofile geladen.")
            else:
                # Standardbenutzer erstellen, wenn keine Profile existieren
                self.create_default_profiles()
                self.save_profiles()
        except Exception as e:
            logger.error(f"Fehler beim Laden von Benutzerprofilen: {e}")
            # Standardbenutzer erstellen
            self.create_default_profiles()
    
    def create_default_profiles(self) -> None:
        """Erstellt Standardbenutzerprofile für Administrator und normalen Benutzer.
        
        Die Standardprofile enthalten vordefinierte Rollen und Berechtigungen.
        """
        self.profiles = {
            "admin": {
                "name": "Administrator",
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "email": "admin@example.com",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "preferences": {},
                "permissions": ["read", "write", "admin", "export", "import"]
            },
            "user": {
                "name": "Standardbenutzer",
                "password_hash": self._hash_password("user123"),
                "role": "user",
                "email": "user@example.com",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "preferences": {},
                "permissions": ["read", "export"]
            }
        }
        logger.info("Standardbenutzerprofile erstellt.")
    
    def save_profiles(self) -> bool:
        """Speichert Benutzerprofile in der Speicherdatei.
        
        Returns:
            bool: True wenn erfolgreich gespeichert, sonst False
        """
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.profiles, f, indent=4, ensure_ascii=False)
            logger.info("Benutzerprofile gespeichert.")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern von Benutzerprofilen: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Erstellt einen sicheren Hash für ein Passwort.
        
        In einer Produktionsumgebung sollte bcrypt anstelle von SHA-256 verwendet werden
        und ein zufälliger Salt pro Benutzer generiert werden.
        
        Args:
            password: Das zu hashende Passwort
            
        Returns:
            str: Der Hash des Passworts
        """
        salt = "streamlit_app_salt"  # In einer echten Anwendung sollte ein zufälliger Salt pro Benutzer verwendet werden
        salted_password = password + salt
        return hashlib.sha256(salted_password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authentifiziert einen Benutzer mit Benutzername und Passwort.
        
        Args:
            username: Der Benutzername
            password: Das Passwort
            
        Returns:
            bool: True wenn Authentifizierung erfolgreich, sonst False
        """
        if username in self.profiles:
            password_hash = self._hash_password(password)
            if password_hash == self.profiles[username]["password_hash"]:
                # Login-Zeit aktualisieren
                self.profiles[username]["last_login"] = datetime.now().isoformat()
                self.save_profiles()
                self.active_user = username
                logger.info(f"Benutzer {username} authentifiziert.")
                return True
        
        logger.warning(f"Fehlgeschlagener Authentifizierungsversuch für Benutzer {username}.")
        return False
    
    def create_user(self, username, password, name, email, role="user", permissions=None):
        """Erstellt einen neuen Benutzer"""
        if username in self.profiles:
            logger.warning(f"Benutzer {username} existiert bereits.")
            return False
        
        if permissions is None:
            permissions = ["read", "export"] if role == "user" else ["read", "write", "admin", "export", "import"]
        
        self.profiles[username] = {
            "name": name,
            "password_hash": self._hash_password(password),
            "role": role,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "preferences": {},
            "permissions": permissions
        }
        
        self.save_profiles()
        logger.info(f"Neuer Benutzer {username} erstellt.")
        return True
    
    def update_user(self, username, **kwargs):
        """Aktualisiert einen Benutzer"""
        if username not in self.profiles:
            logger.warning(f"Benutzer {username} existiert nicht.")
            return False
        
        for key, value in kwargs.items():
            if key == "password":
                self.profiles[username]["password_hash"] = self._hash_password(value)
            elif key in self.profiles[username]:
                self.profiles[username][key] = value
        
        self.save_profiles()
        logger.info(f"Benutzer {username} aktualisiert.")
        return True
    
    def delete_user(self, username):
        """Löscht einen Benutzer"""
        if username not in self.profiles:
            logger.warning(f"Benutzer {username} existiert nicht.")
            return False
        
        del self.profiles[username]
        self.save_profiles()
        logger.info(f"Benutzer {username} gelöscht.")
        return True
    
    def get_user(self, username):
        """Gibt Benutzerdetails zurück"""
        if username in self.profiles:
            # Kopie zurückgeben, um das Original zu schützen
            user_data = dict(self.profiles[username])
            # Passwort-Hash aus Sicherheitsgründen entfernen
            if "password_hash" in user_data:
                del user_data["password_hash"]
            return user_data
        return None
    
    def get_active_user(self):
        """Gibt den aktiven Benutzer zurück"""
        if self.active_user:
            return self.get_user(self.active_user)
        return None
    
    def check_permission(self, permission):
        """Prüft, ob der aktive Benutzer eine bestimmte Berechtigung hat"""
        if not self.active_user:
            return False
        
        user_permissions = self.profiles[self.active_user].get("permissions", [])
        return permission in user_permissions or "admin" in user_permissions
    
    def list_users(self):
        """Listet alle Benutzer auf"""
        user_list = []
        for username, data in self.profiles.items():
            user_data = {
                "username": username,
                "name": data.get("name", ""),
                "role": data.get("role", ""),
                "email": data.get("email", ""),
                "created_at": data.get("created_at", ""),
                "last_login": data.get("last_login", "")
            }
            user_list.append(user_data)
        return user_list
    
    def save_user_preference(self, key, value):
        """Speichert eine Benutzereinstellung"""
        if not self.active_user:
            return False
        
        if "preferences" not in self.profiles[self.active_user]:
            self.profiles[self.active_user]["preferences"] = {}
        
        self.profiles[self.active_user]["preferences"][key] = value
        self.save_profiles()
        return True
    
    def get_user_preference(self, key, default=None):
        """Gibt eine Benutzereinstellung zurück"""
        if not self.active_user:
            return default
        
        preferences = self.profiles[self.active_user].get("preferences", {})
        return preferences.get(key, default)

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
        if isinstance(data, pd.DataFrame):
            # DataFrame als CSV speichern
            data_path = os.path.join(version_dir, "data.csv")
            data.to_csv(data_path, index=False)
        else:
            # Andere Daten als JSON speichern
            data_path = os.path.join(version_dir, "data.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Metadaten speichern
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "version_id": version_id,
            "timestamp": timestamp,
            "created_at": datetime.now().isoformat(),
            "data_type": "dataframe" if isinstance(data, pd.DataFrame) else "json",
            "size": len(data) if isinstance(data, pd.DataFrame) else len(str(data)),
            "name": name
        })
        
        meta_path = os.path.join(version_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Datenversion {version_id} gespeichert")
        return version_id
    
    def load_version(self, version_id):
        """Lädt eine gespeicherte Datenversion"""
        version_dir = os.path.join(self.base_dir, version_id)
        
        if not os.path.exists(version_dir):
            logger.warning(f"Datenversion {version_id} nicht gefunden")
            return None, None
        
        # Metadaten laden
        meta_path = os.path.join(version_dir, "metadata.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Metadaten: {e}")
            return None, None
        
        # Daten basierend auf Typ laden
        data_type = metadata.get("data_type", "json")
        
        try:
            if data_type == "dataframe":
                data_path = os.path.join(version_dir, "data.csv")
                data = pd.read_csv(data_path)
            else:
                data_path = os.path.join(version_dir, "data.json")
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Daten: {e}")
            return None, metadata
        
        logger.info(f"Datenversion {version_id} geladen")
        return data, metadata
    
    def list_versions(self, name=None):
        """Listet alle verfügbaren Versionen auf"""
        versions = []
        
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            
            if os.path.isdir(item_path):
                # Wenn name angegeben ist, nur passende Versionen anzeigen
                if name is not None and not item.startswith(name):
                    continue
                
                meta_path = os.path.join(item_path, "metadata.json")
                
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                        
                        versions.append(metadata)
                    except Exception as e:
                        logger.error(f"Fehler beim Laden der Metadaten für {item}: {e}")
        
        # Nach Zeitstempel sortieren (neueste zuerst)
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return versions
    
    def delete_version(self, version_id):
        """Löscht eine Datenversion"""
        version_dir = os.path.join(self.base_dir, version_id)
        
        if not os.path.exists(version_dir):
            logger.warning(f"Datenversion {version_id} nicht gefunden")
            return False
        
        try:
            # Alle Dateien im Verzeichnis löschen
            for file in os.listdir(version_dir):
                file_path = os.path.join(version_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Verzeichnis löschen
            os.rmdir(version_dir)
            logger.info(f"Datenversion {version_id} gelöscht")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Löschen der Datenversion {version_id}: {e}")
            return False
    
    def compare_versions(self, version_id1, version_id2):
        """Vergleicht zwei Datenversionen"""
        data1, meta1 = self.load_version(version_id1)
        data2, meta2 = self.load_version(version_id2)
        
        if data1 is None or data2 is None:
            return None
        
        comparison = {
            "version1": meta1,
            "version2": meta2,
            "comparison_time": datetime.now().isoformat()
        }
        
        # Wenn beide Daten DataFrames sind
        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            # Anzahl der Zeilen und Spalten vergleichen
            comparison["rows_diff"] = len(data1) - len(data2)
            comparison["columns_diff"] = len(data1.columns) - len(data2.columns)
            
            # Gemeinsame Spalten finden
            common_columns = set(data1.columns).intersection(set(data2.columns))
            comparison["common_columns"] = list(common_columns)
            comparison["unique_columns_1"] = list(set(data1.columns) - set(data2.columns))
            comparison["unique_columns_2"] = list(set(data2.columns) - set(data1.columns))
            
            # Nur wenn die Indizes gleich sind, detaillierten Wertevergleich durchführen
            if len(data1) == len(data2) and all(data1.index == data2.index):
                # Für jede gemeinsame Spalte, Unterschiede zählen
                column_diffs = {}
                for col in common_columns:
                    diff_count = (data1[col] != data2[col]).sum()
                    if diff_count > 0:
                        column_diffs[col] = diff_count
                
                comparison["column_differences"] = column_diffs
        
        # Wenn beide Daten JSON-Objekte sind
        elif isinstance(data1, dict) and isinstance(data2, dict):
            # Schlüssel vergleichen
            keys1 = set(data1.keys())
            keys2 = set(data2.keys())
            
            comparison["common_keys"] = list(keys1.intersection(keys2))
            comparison["unique_keys_1"] = list(keys1 - keys2)
            comparison["unique_keys_2"] = list(keys2 - keys1)
        
        return comparison

class DatabaseManager:
    """Erweiterte Klasse für Datenbankoperationen mit Unterstützung für MongoDB und SQLite"""
    
    def __init__(self, db_type="sqlite", connection_string=None, db_name="kursplan_analyse"):
        self.db_type = db_type.lower()
        self.connection_string = connection_string
        self.db_name = db_name
        self.connection = None
        self.is_connected = False
        
        # Standardwerte für Connection-Strings
        if self.connection_string is None:
            if self.db_type == "sqlite":
                self.connection_string = f"data/{db_name}.db"
            elif self.db_type == "mongodb":
                self.connection_string = f"mongodb://localhost:27017/{db_name}"
    
    def connect(self):
        """Stellt eine Verbindung zur Datenbank her mit erweiterten Fehlerprüfungen"""
        try:
            if self.db_type == "sqlite":
                import sqlite3

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
                import pymongo

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
                return False
                
        except Exception as e:
            logger.error(f"Datenbankfehler: {e}")
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
                import pymongo

                # MongoDB-Datenbank und Collection auswählen
                db = self.connection[self.db_name]
                collection = db[table_name]
                
                # DataFrame in Dictionary-Liste konvertieren
                records = df.to_dict("records")
                
                # Bestehende Daten löschen, wenn gewünscht
                if if_exists == "replace":
                    collection.delete_many({})
                
                # Daten einfügen
                if records:
                    # Batch-Insert für bessere Performance
                    batch_size = 1000
                    for i in range(0, len(records), batch_size):
                        batch = records[i:i + batch_size]
                        collection.insert_many(batch)
                
                logger.info(f"{len(records)} Datensätze in MongoDB-Collection {table_name} gespeichert")
                return True
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return False
        
        except Exception as e:
            logger.error(f"Fehler beim Speichern in der Datenbank: {e}")
            return False
    
    def load_dataframe(self, table_name, query=None, columns=None):
        """Lädt ein DataFrame aus der Datenbank mit erweiterten Filteroptionen"""
        try:
            if not self.is_connected:
                if not self.connect():
                    return pd.DataFrame()
            
            if self.db_type == "sqlite":
                # SQL-Abfrage erstellen
                column_str = ", ".join(columns) if columns else "*"
                sql = f"SELECT {column_str} FROM {table_name}"
                
                # WHERE-Klausel hinzufügen, wenn query vorhanden ist
                params = {}
                if query and isinstance(query, dict):
                    where_clauses = []
                    
                    for key, value in query.items():
                        where_clauses.append(f"{key} = :{key}")
                        params[key] = value
                    
                    if where_clauses:
                        sql += " WHERE " + " AND ".join(where_clauses)
                
                # DataFrame laden
                df = pd.read_sql_query(sql, self.connection, params=params)
                logger.info(f"{len(df)} Datensätze aus SQLite-Tabelle {table_name} geladen")
                return df
            
            elif self.db_type == "mongodb":
                import pymongo

                # MongoDB-Datenbank und Collection auswählen
                db = self.connection[self.db_name]
                collection = db[table_name]
                
                # Abfrage ausführen
                mongo_query = query or {}
                projection = {col: 1 for col in columns} if columns else None
                
                cursor = collection.find(mongo_query, projection)
                df = pd.DataFrame(list(cursor))
                
                # MongoDB-ID entfernen, falls vorhanden und nicht explizit angefordert
                if "_id" in df.columns and (columns is None or "_id" not in columns):
                    df = df.drop("_id", axis=1)
                
                logger.info(f"{len(df)} Datensätze aus MongoDB-Collection {table_name} geladen")
                return df
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Fehler beim Laden aus der Datenbank: {e}")
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
                    results = cursor.fetchall()
                    logger.info(f"SQL-Abfrage ausgeführt: {query[:50]}...")
                    return results
                except Exception:
                    # Kein Ergebnis (z.B. bei INSERT, UPDATE, DELETE)
                    logger.info(f"SQL-Befehl ausgeführt: {query[:50]}...")
                    return cursor.rowcount
            
            elif self.db_type == "mongodb":
                import pymongo

                # MongoDB unterstützt keine SQL-Abfragen
                logger.warning("Benutzerdefinierte Abfragen werden für MongoDB nicht unterstützt")
                return None
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return None
        
        except Exception as e:
            logger.error(f"Fehler bei der Ausführung der Abfrage: {e}")
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
                tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"{len(tables)} Tabellen in SQLite-Datenbank gefunden")
                return tables
            
            elif self.db_type == "mongodb":
                db = self.connection[self.db_name]
                collections = db.list_collection_names()
                logger.info(f"{len(collections)} Collections in MongoDB-Datenbank gefunden")
                return collections
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return []
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Tabellen: {e}")
            return []
    
    def backup_database(self, backup_path=None):
        """Erstellt ein Backup der Datenbank"""
        try:
            if not self.is_connected:
                if not self.connect():
                    return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if backup_path is None:
                if self.db_type == "sqlite":
                    backup_dir = "backups/sqlite"
                    if not os.path.exists(backup_dir):
                        os.makedirs(backup_dir)
                    backup_path = f"{backup_dir}/{os.path.basename(self.connection_string)}_{timestamp}.backup"
                elif self.db_type == "mongodb":
                    backup_dir = "backups/mongodb"
                    if not os.path.exists(backup_dir):
                        os.makedirs(backup_dir)
                    backup_path = f"{backup_dir}/{self.db_name}_{timestamp}"
            
            if self.db_type == "sqlite":
                import shutil

                # Kopiere die Datenbankdatei
                shutil.copy2(self.connection_string, backup_path)
                logger.info(f"SQLite-Datenbank-Backup erstellt: {backup_path}")
                return True
            
            elif self.db_type == "mongodb":
                import subprocess

                # Führe mongodump aus
                command = [
                    "mongodump",
                    "--uri", self.connection_string,
                    "--out", backup_path
                ]
                
                result = subprocess.run(command, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"MongoDB-Datenbank-Backup erstellt: {backup_path}")
                    return True
                else:
                    logger.error(f"Fehler beim Erstellen des MongoDB-Backups: {result.stderr}")
                    return False
            
            else:
                logger.error(f"Nicht unterstützter Datenbanktyp: {self.db_type}")
                return False
        
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Datenbank-Backups: {e}")
            return False

def parallel_process(func, items, max_workers=None, chunk_size=None):
    """Führt eine Funktion parallel für mehrere Elemente aus"""
    if not items:
        return []
    
    # Maximale Anzahl von Workers bestimmen
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Maximal 8 Kerne verwenden
    
    # Bei sehr wenigen Elementen sequentielle Verarbeitung verwenden
    if len(items) <= 2:
        return [func(item) for item in items]
    
    # Aufteilen in Chunks, wenn chunk_size angegeben ist
    if chunk_size is not None and chunk_size > 1:
        chunked_items = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
        # Wrapper-Funktion für Chunk-Verarbeitung
        chunk_func = lambda chunk: [func(item) for item in chunk]
        items_to_process = chunked_items
        process_func = chunk_func
    else:
        items_to_process = items
        process_func = func
    
    try:
        # ThreadPoolExecutor für I/O-gebundene Operationen verwenden
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_func, items_to_process))
        
        # Ergebnisse aus Chunks zusammenführen, falls nötig
        if chunk_size is not None and chunk_size > 1:
            flat_results = []
            for chunk_result in results:
                flat_results.extend(chunk_result)
            return flat_results
        else:
            return results
    
    except Exception as e:
        logger.error(f"Fehler bei der parallelen Verarbeitung: {e}")
        # Fallback zur sequentiellen Verarbeitung
        try:
            return [func(item) for item in items]
        except Exception as e2:
            logger.error(f"Auch sequentielle Verarbeitung fehlgeschlagen: {e2}")
            return []

def process_parallel(data_chunks, process_func, max_workers=None, use_processes=False):
    """Verarbeitet Daten parallel mit fortschrittlicher Fehlerbehandlung und Fortschrittsanzeige"""
    
    if not data_chunks:
        return []
    
    # Maximale Anzahl von Workers bestimmen
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # Maximal 8 Kerne verwenden
    
    # Fortschrittsanzeige initialisieren
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_chunks = len(data_chunks)
    processed_chunks = 0
    results = []
    errors = []
    
    # Shared Queue für Ergebnisse und Fortschritt
    result_queue = queue.Queue()
    
    def process_chunk_with_feedback(chunk, chunk_idx):
        """Verarbeitet einen Chunk und gibt Feedback über die Queue"""
        try:
            result = process_func(chunk)
            result_queue.put(("result", chunk_idx, result))
            return result
        except Exception as e:
            error_msg = f"Fehler bei Chunk {chunk_idx}: {str(e)}"
            result_queue.put(("error", chunk_idx, error_msg))
            return None
    
    def progress_monitor():
        """Überwacht den Fortschritt und aktualisiert die Anzeige"""
        while processed_chunks < total_chunks:
            try:
                msg_type, chunk_idx, data = result_queue.get(timeout=0.1)
                nonlocal processed_chunks
                processed_chunks += 1
                
                if msg_type == "result":
                    results.append((chunk_idx, data))
                elif msg_type == "error":
                    errors.append(data)
                    logger.error(data)
                
                # Fortschritt aktualisieren
                progress = processed_chunks / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"Verarbeitet: {processed_chunks}/{total_chunks} Chunks" + 
                               (f" ({len(errors)} Fehler)" if errors else ""))
                
                result_queue.task_done()
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Fehler im Fortschrittsmonitor: {e}")
    
    # Fortschrittsmonitor-Thread starten
    monitor_thread = threading.Thread(target=progress_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        # Daten parallel verarbeiten
        if use_processes:
            # ProcessPoolExecutor für CPU-gebundene Aufgaben
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                executor_results = []
                for i, chunk in enumerate(data_chunks):
                    future = executor.submit(process_func, chunk)
                    future.add_done_callback(
                        lambda f, idx=i: result_queue.put(
                            ("result", idx, f.result()) if not f.exception() else 
                            ("error", idx, f"Fehler bei Chunk {idx}: {str(f.exception())}")
                        )
                    )
                    executor_results.append(future)
                
                # Warten, bis alle Futures abgeschlossen sind
                for future in executor_results:
                    future.result()
        else:
            # ThreadPoolExecutor für I/O-gebundene Aufgaben
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                executor_results = []
                for i, chunk in enumerate(data_chunks):
                    future = executor.submit(process_chunk_with_feedback, chunk, i)
                    executor_results.append(future)
                
                # Warten, bis alle Futures abgeschlossen sind
                for future in executor_results:
                    future.result()
    
    except Exception as e:
        logger.error(f"Fehler bei der parallelen Verarbeitung: {e}")
        status_text.text(f"Fehler bei der Verarbeitung: {str(e)}")
    
    # Fortschrittsmonitor beenden
    processed_chunks = total_chunks  # Signal zum Beenden des Monitors
    monitor_thread.join(timeout=1.0)
    
    # Ausgabe zusammenstellen
    status_text.text(f"Verarbeitung abgeschlossen: {len(results)} Chunks erfolgreich, {len(errors)} Fehler")
    
    # Ergebnisse sortieren und zurückgeben
    sorted_results = [r[1] for r in sorted(results, key=lambda x: x[0])]
    
    # Fortschrittsanzeige entfernen
    progress_placeholder.empty()
    progress_bar.empty()
    
    return sorted_results, errors

def setup_service_worker():
    """Richtet einen Service Worker für PWA-Funktionalität ein."""
    # Service Worker deaktiviert, um Fehler zu vermeiden
    pass

def detect_device():
    """Erkennt den Gerätetyp des Benutzers."""
    # Vereinfachte Geräteerkennung (ohne JavaScript)
    user_agent = st.session_state.get('user_agent', '')
    
    if any(keyword in user_agent.lower() for keyword in ['android', 'iphone', 'ipod', 'windows phone']):
        return 'mobile'
    elif any(keyword in user_agent.lower() for keyword in ['ipad', 'tablet']):
        return 'tablet'
    else:
        return 'desktop'

def is_mobile():
    """Prüft, ob der Benutzer ein mobiles Gerät verwendet."""
    return detect_device() == 'mobile'

def apply_responsive_styles():
    """Wendet responsive Styles basierend auf dem Gerätetyp an."""
    device_type = detect_device()
    
    # Vereinfachte gerätespezifische Styles
    if device_type == 'mobile':
        # Mobile Styles
        st.markdown("<style>.mobile-content {display: block;}</style>", unsafe_allow_html=True)
    elif device_type == 'tablet':
        # Tablet Styles
        st.markdown("<style>.tablet-content {display: block;}</style>", unsafe_allow_html=True)
    else:
        # Desktop Styles
        st.markdown("<style>.desktop-content {display: block;}</style>", unsafe_allow_html=True)
