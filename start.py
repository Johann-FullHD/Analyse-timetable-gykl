#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ██╗      █████╗ ███╗   ██╗ ║
║   ██║ ██╔╝██║   ██║██╔══██╗██╔════╝██╔══██╗██║     ██╔══██╗████╗  ██║ ║
║   █████╔╝ ██║   ██║██████╔╝███████╗██████╔╝██║     ███████║██╔██╗ ██║ ║
║   ██╔═██╗ ██║   ██║██╔══██╗╚════██║██╔═══╝ ██║     ██╔══██║██║╚██╗██║ ║
║   ██║  ██╗╚██████╔╝██║  ██║███████║██║     ███████╗██║  ██║██║ ╚████║ ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ║
║                                                                       ║
║   𝓐𝓷𝓪𝓵𝔂𝓼𝓮                                                   v2.0.0  ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

Startdatei für die Kursplan-Analyse-Anwendung
=============================================

Diese Datei dient als Einstiegspunkt für die Kursplan-Analyse-Anwendung.
Sie prüft die Umgebung, stellt sicher, dass alle erforderlichen Abhängigkeiten 
installiert sind, und startet dann die Hauptanwendung.

Verwendung:
-----------
python start.py [--debug] [--port PORT] [--browser BROWSER] [--theme THEME]

Parameter:
-----------
--debug:   Aktiviert den Debug-Modus (Standard: False)
--port:    Gibt den Port an, auf dem die Anwendung laufen soll (Standard: 8501)
--browser: Gibt an, ob ein Browser automatisch geöffnet werden soll (Standard: 'new')
--theme:   Wählt ein Design-Theme für die Anwendung (Standard: 'modern')
--turbo:   Aktiviert Leistungsoptimierungen (Standard: False)
"""
import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Farbige Terminal-Ausgabe für Windows und Unix
try:
    import colorama
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# Animationen für Terminalausgabe (wenn verfügbar)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn)
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Logging-Konfiguration
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
        "logs/start.log", 
        mode='a', 
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=3,  # 3 Backup-Dateien
        encoding='utf-8'
    )
except ImportError:
    # Fallback, wenn RotatingFileHandler nicht verfügbar ist
    log_handler = logging.FileHandler(
        "logs/start.log", 
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

# Projektinformationen
APP_NAME = "Kursplan-Analyse"
APP_VERSION = "2.0.0"
MAIN_FILE = "app.py"
REQUIREMENTS_FILE = "requirements-core.txt"
OPTIONAL_REQUIREMENTS = "requirements-optional.txt"

# Farbcodes für Terminalausgabe
class Colors:
    HEADER = '\033[95m' if not HAS_COLORAMA else colorama.Fore.MAGENTA
    BLUE = '\033[94m' if not HAS_COLORAMA else colorama.Fore.BLUE
    CYAN = '\033[96m' if not HAS_COLORAMA else colorama.Fore.CYAN
    GREEN = '\033[92m' if not HAS_COLORAMA else colorama.Fore.GREEN
    WARNING = '\033[93m' if not HAS_COLORAMA else colorama.Fore.YELLOW
    FAIL = '\033[91m' if not HAS_COLORAMA else colorama.Fore.RED
    BOLD = '\033[1m' if not HAS_COLORAMA else colorama.Style.BRIGHT
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m' if not HAS_COLORAMA else colorama.Style.RESET_ALL


# Globale Einstellungen
class Settings:
    """Speichert globale Einstellungen für die Anwendung."""
    
    def __init__(self):
        self.debug = False
        self.port = 8501
        self.browser = "new"
        self.theme = "modern"
        self.turbo = False
        self.config_file = Path("config.json")
        self.load_config()
    
    def load_config(self):
        """Lädt gespeicherte Konfiguration, wenn vorhanden."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.debug = config.get('debug', self.debug)
                    self.port = config.get('port', self.port)
                    self.browser = config.get('browser', self.browser)
                    self.theme = config.get('theme', self.theme)
                    self.turbo = config.get('turbo', self.turbo)
                    logger.info(f"Konfiguration aus {self.config_file} geladen")
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Konfiguration: {e}")
    
    def save_config(self):
        """Speichert aktuelle Konfiguration."""
        try:
            config = {
                'debug': self.debug,
                'port': self.port,
                'browser': self.browser,
                'theme': self.theme,
                'turbo': self.turbo
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Konfiguration in {self.config_file} gespeichert")
        except Exception as e:
            logger.warning(f"Fehler beim Speichern der Konfiguration: {e}")


# Initialisiere globale Einstellungen
settings = Settings()

# Rich-Konsole initialisieren, wenn verfügbar
console = Console() if HAS_RICH else None


def fancy_print(message: str, style: str = "info", emoji: str = ""):
    """Gibt formatierten Text aus, mit Rich oder Farben."""
    style_map = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "warning": Colors.WARNING,
        "error": Colors.FAIL,
        "header": Colors.HEADER + Colors.BOLD,
    }
    
    emoji_map = {
        "info": "ℹ️ ",
        "success": "✅ ",
        "warning": "⚠️ ",
        "error": "❌ ",
        "header": "🚀 ",
        "check": "✓ ",
        "loading": "⏳ ",
        "config": "⚙️ ",
        "rocket": "🚀 ",
        "sparkles": "✨ ",
        "fire": "🔥 ",
        "zap": "⚡ ",
        "gear": "⚙️ ",
        "magic": "✨ ",
        "python": "🐍 "
    }
    
    emoji_prefix = emoji_map.get(emoji, emoji) if emoji else ""
    
    if HAS_RICH and console:
        message_style = {
            "info": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red bold",
            "header": "magenta bold",
        }.get(style, "")
        console.print(f"{emoji_prefix}{message}", style=message_style)
    else:
        color = style_map.get(style, Colors.ENDC)
        print(f"{color}{emoji_prefix}{message}{Colors.ENDC}")


def animated_loading(message: str, duration: float = 0.5, cycles: int = 3):
    """Zeigt eine Animation während des Ladens an."""
    if HAS_RICH and console:
        with console.status(message, spinner="dots") as status:
            time.sleep(duration)
    else:
        chars = "|/-\\"
        for _ in range(cycles * len(chars)):
            time.sleep(duration / (cycles * len(chars)))
            print(f"\r{message} {chars[_ % len(chars)]}", end="")
        print()


def check_system_compatibility() -> Tuple[bool, List[str]]:
    """Überprüft, ob das System für die Anwendung kompatibel ist."""
    issues = []
    compatible = True
    
    # Minimale Systemanforderungen
    min_ram_mb = 2048  # 2 GB RAM
    min_disk_space_mb = 500  # 500 MB freier Speicherplatz
    
    fancy_print("Überprüfe Systemkompatibilität...", "info", "check")
    
    # Betriebssystem prüfen
    os_name = platform.system()
    os_version = platform.release()
    fancy_print(f"Betriebssystem: {os_name} {os_version}", "info")
    
    if os_name not in ["Windows", "Linux", "Darwin"]:
        issues.append(f"Nicht unterstütztes Betriebssystem: {os_name}")
        compatible = False
    
    # CPU prüfen
    cpu_count = os.cpu_count() or 0
    if cpu_count < 2:
        issues.append(f"Zu wenig CPU-Kerne: {cpu_count} (mindestens 2 empfohlen)")
        compatible = False
    
    # Verfügbaren Arbeitsspeicher prüfen (wenn möglich)
    try:
        if os_name == "Windows":
            import psutil
            ram_mb = psutil.virtual_memory().total / (1024 * 1024)
            free_disk_mb = psutil.disk_usage('/').free / (1024 * 1024)
            
            if ram_mb < min_ram_mb:
                issues.append(f"Zu wenig RAM: {ram_mb:.0f} MB (mindestens {min_ram_mb} MB empfohlen)")
                compatible = False
            
            if free_disk_mb < min_disk_space_mb:
                issues.append(f"Zu wenig freier Speicherplatz: {free_disk_mb:.0f} MB (mindestens {min_disk_space_mb} MB erforderlich)")
                compatible = False
    except ImportError:
        pass  # psutil nicht verfügbar, Speicherprüfung überspringen
    
    return compatible, issues


def check_python_version() -> bool:
    """Überprüft, ob die Python-Version kompatibel ist."""
    required_version = (3, 9)
    current_version = sys.version_info[:2]
    python_impl = platform.python_implementation()
    
    fancy_print("Überprüfe Python-Version...", "info", "python")
    fancy_print(f"Python {python_impl} {'.'.join(map(str, current_version))}", "info")
    
    if current_version < required_version:
        fancy_print(
            f"Python {required_version[0]}.{required_version[1]} oder höher erforderlich. "
            f"Aktuelle Version: {current_version[0]}.{current_version[1]}", 
            "error"
        )
        fancy_print("Bitte installieren Sie eine neuere Python-Version: https://www.python.org/downloads/", "error")
        return False
    
    fancy_print(f"Python-Version {current_version[0]}.{current_version[1]} ist kompatibel ✓", "success")
    return True


def is_package_installed(package_name: str) -> bool:
    """Prüft, ob ein Paket installiert ist."""
    try:
        importlib.util.find_spec(package_name.split('==')[0].strip())
        return True
    except ImportError:
        return False


def check_dependencies(install_missing: bool = True) -> bool:
    """Überprüft, ob alle erforderlichen Abhängigkeiten installiert sind."""
    fancy_print("Prüfe Abhängigkeiten...", "info", "gear")
    
    # Überprüfen, ob die requirements.txt existiert
    req_file = Path(REQUIREMENTS_FILE)
    if not req_file.exists():
        fancy_print(f"Die Datei {REQUIREMENTS_FILE} wurde nicht gefunden.", "error")
        # Erstelle eine minimale requirements.txt mit den notwendigsten Paketen
        try:
            with open(req_file, 'w', encoding='utf-8') as f:
                f.write("""# Minimale Abhängigkeiten
streamlit>=1.20.0
pandas>=1.0.0
numpy>=1.20.0
matplotlib>=3.0.0
""")
            fancy_print(f"Minimale {REQUIREMENTS_FILE} wurde erstellt.", "success")
        except Exception as e:
            fancy_print(f"Fehler beim Erstellen der minimalen {REQUIREMENTS_FILE}: {e}", "error")
            return False
    
    # Prüfe, ob Streamlit installiert ist
    if not is_package_installed("streamlit"):
        fancy_print("Streamlit ist nicht installiert.", "warning")
        if install_missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit>=1.20.0"])
                fancy_print("Streamlit wurde installiert.", "success")
            except Exception as e:
                fancy_print(f"Fehler beim Installieren von Streamlit: {e}", "error")
                return False
        else:
            fancy_print("Bitte installieren Sie Streamlit: pip install streamlit", "info")
            return False
    
    # Paketliste aus requirements.txt lesen
    with open(req_file, 'r', encoding='utf-8') as f:
        required_packages = [
            line.strip() for line in f.readlines() 
            if line.strip() and not line.startswith('#')
        ]
    
    # Prüfen, ob alle erforderlichen Pakete installiert sind
    missing_packages = []
    
    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[green]{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("Prüfe Pakete...", total=len(required_packages))
            
            for package in required_packages:
                package_name = package.split('==')[0].strip()
                if not is_package_installed(package_name):
                    missing_packages.append(package)
                progress.update(task, advance=1)
    else:
        for i, package in enumerate(required_packages):
            package_name = package.split('==')[0].strip()
            print(f"\rPrüfe Paket {i+1}/{len(required_packages)}: {package_name}", end="")
            if not is_package_installed(package_name):
                missing_packages.append(package)
        print()
    
    # Wenn Pakete fehlen, installieren oder warnen
    if missing_packages:
        fancy_print(f"{len(missing_packages)} von {len(required_packages)} Paketen fehlen:", "warning")
        for package in missing_packages[:5]:
            fancy_print(f"  - {package}", "warning")
        if len(missing_packages) > 5:
            fancy_print(f"  - ... und {len(missing_packages)-5} weitere", "warning")
        
        if install_missing:
            return install_dependencies(missing_packages)
        else:
            fancy_print("Bitte installieren Sie die fehlenden Abhängigkeiten:", "warning")
            fancy_print(f"pip install -r {REQUIREMENTS_FILE}", "info")
            return False
    
    fancy_print("Alle erforderlichen Pakete sind installiert ✓", "success")
    return True


def install_dependencies(packages: Optional[List[str]] = None) -> bool:
    """Installiert fehlende Abhängigkeiten."""
    fancy_print("Installiere Abhängigkeiten...", "info", "loading")
    
    # Wenn keine spezifischen Pakete angegeben sind, alle aus requirements.txt installieren
    if packages is None:
        cmd = [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE]
        fancy_print(f"Installiere alle Pakete aus {REQUIREMENTS_FILE}...", "info")
    else:
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        fancy_print(f"Installiere {len(packages)} fehlende Pakete...", "info")
    
    # Fortgeschrittene Animation mit Rich wenn verfügbar
    if HAS_RICH and console:
        with console.status("[bold green]Installiere Pakete...", spinner="dots"):
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                console.print(Panel(Text("Abhängigkeiten erfolgreich installiert", style="green bold")))
                return True
            except subprocess.CalledProcessError as e:
                console.print(Panel(Text(f"Fehler bei der Installation: {e}", style="red bold")))
                console.print(Text(e.stderr, style="red"))
                return False
    else:
        try:
            subprocess.check_call(cmd)
            fancy_print("Abhängigkeiten erfolgreich installiert", "success")
            return True
        except subprocess.CalledProcessError as e:
            fancy_print(f"Fehler bei der Installation der Abhängigkeiten: {e}", "error")
            return False


def setup_environment() -> bool:
    """Richtet die Umgebung für die Anwendung ein."""
    fancy_print("Richte Umgebung ein...", "info", "gear")
    
    # Erstellen von erforderlichen Verzeichnissen
    required_dirs = ["data", "logs", "temp", "reports", "cache"]
    
    if HAS_RICH and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Erstelle Verzeichnisse...", total=len(required_dirs))
            
            for directory in required_dirs:
                dir_path = Path(directory)
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                progress.update(task, advance=1, description=f"Verzeichnis '{directory}'")
    else:
        for directory in required_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                fancy_print(f"Verzeichnis '{directory}' erstellt", "info")
    
    # Setzen von Umgebungsvariablen
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["KURSPLAN_APP_VERSION"] = APP_VERSION
    os.environ["KURSPLAN_DEBUG"] = str(settings.debug).lower()
    os.environ["KURSPLAN_THEME"] = settings.theme
    
    # Optional: Erstellen einer leeren .streamlit/config.toml falls nicht vorhanden
    streamlit_config_dir = Path(".streamlit")
    streamlit_config_file = streamlit_config_dir / "config.toml"
    
    if not streamlit_config_dir.exists():
        streamlit_config_dir.mkdir(parents=True, exist_ok=True)
    
    if not streamlit_config_file.exists():
        with open(streamlit_config_file, 'w', encoding='utf-8') as f:
            f.write(f"""[theme]
primaryColor = "#2563EB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
textColor = "#262730"
font = "sans serif"

[server]
port = {settings.port}
headless = {"true" if settings.browser == "none" else "false"}
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[logger]
level = {"debug" if settings.debug else "info"}
""")
        fancy_print("Streamlit-Konfiguration erstellt", "info")
    
    fancy_print("Umgebung erfolgreich eingerichtet ✓", "success")
    return True


def fix_app_errors() -> bool:
    """Behebt bekannte Fehler in der Anwendung."""
    fancy_print("Prüfe auf bekannte Probleme...", "info", "magic")
    
    app_path = Path(MAIN_FILE)
    if not app_path.exists():
        fancy_print(f"Die Datei {MAIN_FILE} wurde nicht gefunden.", "error")
        return False
    
    try:
        # Lesen der app.py
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        fixed_something = False
        
        # Beheben des Tab-Problems
        if "tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([" in content:
            content = content.replace(
                "tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([",
                "tab1, tab2, tab3, tab4, tab5 = st.tabs(["
            )
            fixed_something = True
            fancy_print("✓ Tab-Problem in app.py behoben", "success")
        
        # Weitere bekannte Probleme hier beheben...
        # Beispiel: Fehlende Importe ergänzen
        if "import numpy as np" not in content and "import numpy" not in content and "np." in content:
            # Am Anfang nach den Imports einfügen
            import_section_end = content.find("\n\n", content.find("import "))
            if import_section_end > 0:
                content = content[:import_section_end] + "\nimport numpy as np" + content[import_section_end:]
                fixed_something = True
                fancy_print("✓ Fehlender numpy-Import in app.py ergänzt", "success")
        
        # Schreiben der korrigierten app.py, falls Änderungen vorgenommen wurden
        if fixed_something:
            # Backup erstellen
            backup_path = Path("backup") / f"{MAIN_FILE}.bak.{int(time.time())}"
            os.makedirs(backup_path.parent, exist_ok=True)
            shutil.copy2(app_path, backup_path)
            fancy_print(f"Backup erstellt: {backup_path}", "info")
            
            # Korrigierte Datei schreiben
            with open(app_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            fancy_print("✓ Bekannte Probleme in app.py wurden behoben", "success")
        else:
            fancy_print("✓ Keine bekannten Probleme in app.py gefunden", "success")
        
        return True
    except Exception as e:
        fancy_print(f"Fehler beim Beheben von App-Fehlern: {e}", "error")
        return False


def check_optional_enhancements():
    """Prüft und installiert optionale Verbesserungen für die Anwendung."""
    optional_req_file = Path(OPTIONAL_REQUIREMENTS)
    
    if not optional_req_file.exists():
        # Erstelle eine Datei mit optionalen Verbesserungen
        with open(optional_req_file, 'w', encoding='utf-8') as f:
            f.write("""# Optionale Pakete für erweiterte Funktionen
# UI-Verbesserungen
rich>=10.0.0
colorama>=0.4.4
# Leistungsverbesserungen
psutil>=5.9.0
# Erweiterte Datenanalyse
scikit-learn>=1.0.0
plotly>=5.5.0
# Bildverarbeitung
pillow>=9.0.0
""")
    
    fancy_print("Prüfe auf optionale Verbesserungen...", "info", "sparkles")
    
    if HAS_RICH:
        answer = console.input("[blue]Möchten Sie optionale Pakete für zusätzliche Funktionen installieren? [y/N]: [/blue]")
    else:
        answer = input(f"{Colors.BLUE}Möchten Sie optionale Pakete für zusätzliche Funktionen installieren? [y/N]: {Colors.ENDC}")
    
    if answer.lower() in ['y', 'j', 'yes', 'ja']:
        try:
            fancy_print("Installiere optionale Pakete...", "info")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", OPTIONAL_REQUIREMENTS])
            fancy_print("Optionale Pakete erfolgreich installiert ✓", "success")
            return True
        except subprocess.CalledProcessError as e:
            fancy_print(f"Fehler bei der Installation optionaler Pakete: {e}", "warning")
            return False
    else:
        fancy_print("Installation optionaler Pakete übersprungen", "info")
        return True


def run_application() -> bool:
    """Startet die Hauptanwendung."""
    fancy_print(f"Starte {APP_NAME} v{APP_VERSION}...", "header", "rocket")
    
    # Überprüfen, ob die Hauptdatei existiert
    main_file = Path(MAIN_FILE)
    if not main_file.exists():
        fancy_print(f"Die Datei {MAIN_FILE} wurde nicht gefunden.", "error")
        return False
    
    # Streamlit-Befehl vorbereiten
    cmd = [
        sys.executable, "-m", "streamlit", "run", MAIN_FILE,
        "--server.port", str(settings.port)
    ]
    
    if settings.debug:
        cmd.append("--logger.level=debug")
    
    if settings.browser == "none":
        cmd.append("--server.headless=true")
    
    # Theme-Einstellungen
    if settings.theme != "default":
        # Setze Theme-Umgebungsvariable für die App
        os.environ["STREAMLIT_THEME"] = settings.theme
    
    # Turbo-Modus für bessere Performance
    if settings.turbo:
        cmd.append("--server.maxUploadSize=50")
        cmd.append("--server.enableCORS=false")
        cmd.append("--server.enableXsrfProtection=false")
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # App-URL
    app_url = f"http://localhost:{settings.port}"
    
    # Browser automatisch öffnen, wenn gewünscht
    if settings.browser != "none":
        threading.Timer(
            2.0,  # Warte 2 Sekunden, damit Streamlit starten kann
            lambda: webbrowser.open(app_url, new=(settings.browser == "new"))
        ).start()
    
    # Anwendung starten mit cooler Animation
    if HAS_RICH and console:
        console.print(Panel.fit(
            Text(f"{APP_NAME} v{APP_VERSION}", style="bold blue"),
            title="[bold green]Anwendung wird gestartet[/bold green]",
            subtitle=f"[cyan]{app_url}[/cyan]"
        ))
        
        with console.status("[bold green]Starte Anwendung...", spinner="dots"):
            try:
                # Startbefehl anzeigen
                console.print(f"[dim]{' '.join(cmd)}[/dim]")
                # Anwendung starten
                subprocess.run(cmd, check=True)
                return True
            except KeyboardInterrupt:
                console.print("\n[yellow]Anwendung durch Benutzer beendet[/yellow]")
                return True
            except Exception as e:
                console.print(f"[red bold]Fehler beim Starten der Anwendung: {e}[/red bold]")
                return False
    else:
        # Fallback für Terminals ohne Rich
        fancy_print(f"Die Anwendung wird unter {app_url} verfügbar sein.", "info")
        
        try:
            fancy_print(f"Führe Befehl aus: {' '.join(cmd)}", "info")
            subprocess.run(cmd, check=True)
            return True
        except KeyboardInterrupt:
            fancy_print("\nAnwendung durch Benutzer beendet", "warning")
            return True
        except Exception as e:
            fancy_print(f"Fehler beim Starten der Anwendung: {e}", "error")
            return False


def parse_arguments():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(
        description=f'{APP_NAME} v{APP_VERSION}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python start.py                     # Startet mit Standardeinstellungen
  python start.py --debug --port 8502 # Startet im Debug-Modus auf Port 8502
  python start.py --theme dark        # Startet mit dunklem Design
  python start.py --turbo             # Startet mit Leistungsoptimierungen
        """
    )
    
    parser.add_argument('--debug', action='store_true', help='Aktiviert den Debug-Modus')
    parser.add_argument('--port', type=int, default=settings.port, help=f'Port für die Anwendung (Standard: {settings.port})')
    parser.add_argument('--browser', choices=['new', 'existing', 'none'], default=settings.browser,
                      help='Browser-Verhalten: "new" öffnet neuen Tab, "existing" nutzt existierenden, "none" öffnet keinen Browser')
    parser.add_argument('--theme', choices=['modern', 'dark', 'light', 'minimal', 'default'], default=settings.theme,
                      help='Design-Theme für die Anwendung')
    parser.add_argument('--turbo', action='store_true', help='Aktiviert Leistungsoptimierungen')
    parser.add_argument('--check-only', action='store_true', help='Prüft nur die Umgebung ohne die App zu starten')
    parser.add_argument('--fix', action='store_true', help='Behebt bekannte Probleme ohne Nachfrage')
    
    args = parser.parse_args()
    
    # Einstellungen aktualisieren
    settings.debug = args.debug
    settings.port = args.port
    settings.browser = args.browser
    settings.theme = args.theme
    settings.turbo = args.turbo
    
    # Konfiguration speichern
    settings.save_config()
    
    return args


def display_welcome_message():
    """Zeigt eine coole Willkommensnachricht an."""
    system_info = f"System: {platform.system()} {platform.release()}"
    python_info = f"Python: {platform.python_version()}"
    cpu_info = f"CPU: {os.cpu_count()} Kerne"
    
    if HAS_RICH and console:
        # Coole ASCII-Art mit Rich
        console.print("[bold blue]" + "=" * 70 + "[/bold blue]")
        console.print("[bold blue]" + f"{APP_NAME} v{APP_VERSION}".center(70) + "[/bold blue]")
        console.print("[bold blue]" + "=" * 70 + "[/bold blue]")
        
        console.print(Panel(
            Text.from_markup(f"""[bold green]{system_info}[/bold green]
[bold yellow]{python_info}[/bold yellow]
[bold cyan]{cpu_info}[/bold cyan]

[bold magenta]Diese Anwendung ermöglicht die Analyse von Kursplänen und
Stundenplaninformationen für Bildungseinrichtungen.[/bold magenta]

[dim]Entwickelt von: Entwicklerteam Kursplan-Analyse[/dim]
[dim]Lizenz: MIT[/dim]"""),
            title="[bold]Willkommen![/bold]",
            border_style="blue",
            padding=(1, 2)
        ))
    else:
        # Fallback für Terminals ohne Rich
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}{Colors.HEADER}{APP_NAME} v{APP_VERSION}{Colors.ENDC}".center(70))
        print("=" * 70)
        print(f"  {Colors.BLUE}{system_info}{Colors.ENDC}")
        print(f"  {Colors.GREEN}{python_info}{Colors.ENDC}")
        print(f"  {Colors.CYAN}{cpu_info}{Colors.ENDC}")
        print("-" * 70)
        print(f"  {Colors.BOLD}Diese Anwendung ermöglicht die Analyse von Kursplänen und{Colors.ENDC}")
        print(f"  {Colors.BOLD}Stundenplaninformationen für Bildungseinrichtungen.{Colors.ENDC}")
        print("-" * 70)
        print(f"  Autor: Entwicklerteam Kursplan-Analyse")
        print(f"  Lizenz: MIT")
        print("=" * 70 + "\n")


def main():
    """Hauptfunktion zum Starten der Anwendung."""
    # Verzeichnisstruktur für Logs erstellen
    os.makedirs("logs", exist_ok=True)
    
    # Willkommensnachricht anzeigen
    display_welcome_message()
    
    # Kommandozeilenargumente parsen
    args = parse_arguments()
    
    # Systemkompatibilität prüfen
    system_compatible, issues = check_system_compatibility()
    if not system_compatible:
        fancy_print("Systemprüfung fehlgeschlagen:", "warning")
        for issue in issues:
            fancy_print(f"  - {issue}", "warning")
        if not args.fix:
            if HAS_RICH and console:
                continue_anyway = console.input("[yellow]Trotzdem fortfahren? [y/N]: [/yellow]").lower() in ['y', 'j', 'yes', 'ja']
            else:
                continue_anyway = input(f"{Colors.WARNING}Trotzdem fortfahren? [y/N]: {Colors.ENDC}").lower() in ['y', 'j', 'yes', 'ja']
            
            if not continue_anyway:
                return 1
    
    # Umgebung überprüfen und einrichten
    if not check_python_version():
        return 1
    
    if not check_dependencies(install_missing=args.fix):
        return 1
    
    if not setup_environment():
        return 1
    
    # Bekannte Fehler beheben
    if args.fix or args.check_only:
        fix_app_errors()
    else:
        # Interaktive Abfrage
        if HAS_RICH and console:
            fix_issues = console.input("[blue]Bekannte Probleme automatisch beheben? [Y/n]: [/blue]").lower() not in ['n', 'no', 'nein']
        else:
            fix_issues = input(f"{Colors.BLUE}Bekannte Probleme automatisch beheben? [Y/n]: {Colors.ENDC}").lower() not in ['n', 'no', 'nein']
        
        if fix_issues:
            fix_app_errors()
    
    # Optionale Verbesserungen prüfen
    if not args.check_only and not args.fix:
        check_optional_enhancements()
    
    # Bei --check-only Modus hier beenden
    if args.check_only:
        fancy_print("Umgebungsprüfung abgeschlossen. App wird nicht gestartet.", "success")
        return 0
    
    # Anwendung starten
    app_started = run_application()
    return 0 if app_started else 1


def show_menu():
    """Zeigt ein interaktives Menü mit verschiedenen Optionen an."""
    # Speichern des letzten Startzustands, um ihn später wiederholen zu können
    last_launch = {
        'occurred': False,
        'settings': {
            'debug': settings.debug,
            'port': settings.port,
            'browser': settings.browser,
            'theme': settings.theme,
            'turbo': settings.turbo
        }
    }
    
    while True:
        if HAS_RICH and console:
            console.print("\n[bold blue]" + "=" * 50 + "[/bold blue]")
            console.print("[bold cyan]Hauptmenü - Kursplan-Analyse[/bold cyan]")
            console.print("[bold blue]" + "=" * 50 + "[/bold blue]")
            console.print("[bold white]1.[/bold white] Anwendung starten")
            console.print("[bold white]2.[/bold white] Schnellstart (ohne Checks)")
            if last_launch['occurred']:
                console.print("[bold white]3.[/bold white] Letzten Start wiederholen")
            console.print("[bold white]4.[/bold white] Umgebung überprüfen")
            console.print("[bold white]5.[/bold white] Abhängigkeiten installieren")
            console.print("[bold white]6.[/bold white] Fehler beheben")
            console.print("[bold white]7.[/bold white] Einstellungen anpassen")
            console.print("[bold white]8.[/bold white] Hilfe anzeigen")
            console.print("[bold white]0.[/bold white] Beenden")
            console.print("[bold blue]" + "=" * 50 + "[/bold blue]")
            
            choice = console.input("[bold green]Bitte wählen Sie eine Option: [/bold green]")
        else:
            print("\n" + "=" * 50)
            print(f"{Colors.BOLD}{Colors.CYAN}Hauptmenü - Kursplan-Analyse{Colors.ENDC}")
            print("=" * 50)
            print("1. Anwendung starten")
            print("2. Schnellstart (ohne Checks)")
            if last_launch['occurred']:
                print("3. Letzten Start wiederholen")
            print("4. Umgebung überprüfen")
            print("5. Abhängigkeiten installieren")
            print("6. Fehler beheben")
            print("7. Einstellungen anpassen")
            print("8. Hilfe anzeigen")
            print("0. Beenden")
            print("=" * 50)
            
            choice = input(f"{Colors.GREEN}Bitte wählen Sie eine Option: {Colors.ENDC}")
        
        if choice == "1":
            # Anwendung starten
            fancy_print("Starte die Anwendung...", "info", "rocket")
            # Checks durchführen
            check_system_compatibility()
            check_python_version()
            check_dependencies(install_missing=False)
            setup_environment()
            # App starten
            run_application()
            # Startzustand speichern
            last_launch['occurred'] = True
        elif choice == "2":
            # Schnellstart ohne Checks
            fancy_print("Schnellstart ohne Überprüfungen...", "info", "zap")
            run_application()
            # Startzustand speichern
            last_launch['occurred'] = True
        elif choice == "3" and last_launch['occurred']:
            # Letzten Start wiederholen
            fancy_print("Wiederhole letzten Start...", "info", "rocket")
            # Temporär Einstellungen wiederherstellen
            orig_settings = {
                'debug': settings.debug,
                'port': settings.port,
                'browser': settings.browser,
                'theme': settings.theme,
                'turbo': settings.turbo
            }
            
            # Einstellungen des letzten Starts anwenden
            settings.debug = last_launch['settings']['debug']
            settings.port = last_launch['settings']['port']
            settings.browser = last_launch['settings']['browser']
            settings.theme = last_launch['settings']['theme']
            settings.turbo = last_launch['settings']['turbo']
            
            # App starten
            run_application()
            
            # Ursprüngliche Einstellungen wiederherstellen
            settings.debug = orig_settings['debug']
            settings.port = orig_settings['port']
            settings.browser = orig_settings['browser']
            settings.theme = orig_settings['theme']
            settings.turbo = orig_settings['turbo']
        elif choice == "4":
            # Umgebung überprüfen
            fancy_print("Überprüfe die Umgebung...", "info", "check")
            check_system_compatibility()
            check_python_version()
            check_dependencies(install_missing=False)
            setup_environment()
        elif choice == "5":
            # Abhängigkeiten installieren
            fancy_print("Installiere Abhängigkeiten...", "info", "gear")
            install_option = ""
            if HAS_RICH and console:
                install_option = console.input("[blue]Was möchten Sie installieren? (1: Erforderliche Pakete, 2: Optionale Pakete, 3: Alles): [/blue]")
            else:
                install_option = input(f"{Colors.BLUE}Was möchten Sie installieren? (1: Erforderliche Pakete, 2: Optionale Pakete, 3: Alles): {Colors.ENDC}")
            
            if install_option in ["1", "3"]:
                check_dependencies(install_missing=True)
            if install_option in ["2", "3"]:
                check_optional_enhancements()
        elif choice == "6":
            # Fehler beheben
            fancy_print("Behebe bekannte Fehler...", "info", "magic")
            fix_app_errors()
        elif choice == "7":
            # Einstellungen anpassen
            fancy_print("Einstellungen anpassen...", "info", "config")
            configure_settings()
        elif choice == "8":
            # Hilfe anzeigen
            show_help()
        elif choice == "0":
            # Beenden
            fancy_print("Programm wird beendet...", "info")
            return
        else:
            fancy_print(f"Ungültige Eingabe: {choice}", "error")


def configure_settings():
    """Erlaubt dem Benutzer, die Einstellungen anzupassen."""
    if HAS_RICH and console:
        console.print("\n[bold blue]" + "=" * 50 + "[/bold blue]")
        console.print("[bold cyan]Einstellungen - Kursplan-Analyse[/bold cyan]")
        console.print("[bold blue]" + "=" * 50 + "[/bold blue]")
        console.print(f"[bold white]1.[/bold white] Debug-Modus: [{'Ein' if settings.debug else 'Aus'}]")
        console.print(f"[bold white]2.[/bold white] Port: [{settings.port}]")
        console.print(f"[bold white]3.[/bold white] Browser: [{settings.browser}]")
        console.print(f"[bold white]4.[/bold white] Theme: [{settings.theme}]")
        console.print(f"[bold white]5.[/bold white] Turbo-Modus: [{'Ein' if settings.turbo else 'Aus'}]")
        console.print("[bold white]0.[/bold white] Zurück zum Hauptmenü")
        console.print("[bold blue]" + "=" * 50 + "[/bold blue]")
        
        choice = console.input("[bold green]Bitte wählen Sie eine Option: [/bold green]")
    else:
        print("\n" + "=" * 50)
        print(f"{Colors.BOLD}{Colors.CYAN}Einstellungen - Kursplan-Analyse{Colors.ENDC}")
        print("=" * 50)
        print(f"1. Debug-Modus: [{'Ein' if settings.debug else 'Aus'}]")
        print(f"2. Port: [{settings.port}]")
        print(f"3. Browser: [{settings.browser}]")
        print(f"4. Theme: [{settings.theme}]")
        print(f"5. Turbo-Modus: [{'Ein' if settings.turbo else 'Aus'}]")
        print("0. Zurück zum Hauptmenü")
        print("=" * 50)
        
        choice = input(f"{Colors.GREEN}Bitte wählen Sie eine Option: {Colors.ENDC}")
    
    if choice == "1":
        # Debug-Modus ändern
        settings.debug = not settings.debug
        fancy_print(f"Debug-Modus: {'Ein' if settings.debug else 'Aus'}", "success")
    elif choice == "2":
        # Port ändern
        if HAS_RICH and console:
            port_input = console.input("[blue]Neuer Port (1024-65535): [/blue]")
        else:
            port_input = input(f"{Colors.BLUE}Neuer Port (1024-65535): {Colors.ENDC}")
        
        try:
            port = int(port_input)
            if 1024 <= port <= 65535:
                settings.port = port
                fancy_print(f"Port geändert auf: {port}", "success")
            else:
                fancy_print("Ungültiger Port. Bitte wählen Sie einen Wert zwischen 1024 und 65535.", "error")
        except ValueError:
            fancy_print("Ungültige Eingabe. Bitte geben Sie eine Zahl ein.", "error")
    elif choice == "3":
        # Browser-Einstellung ändern
        if HAS_RICH and console:
            console.print("[bold]Browser-Optionen:[/bold]")
            console.print("  [dim]new:[/dim] Öffnet einen neuen Browser-Tab")
            console.print("  [dim]existing:[/dim] Verwendet existierenden Tab, falls möglich")
            console.print("  [dim]none:[/dim] Öffnet keinen Browser automatisch")
            browser_input = console.input("[blue]Neue Browser-Einstellung (new/existing/none): [/blue]")
        else:
            print("Browser-Optionen:")
            print("  new: Öffnet einen neuen Browser-Tab")
            print("  existing: Verwendet existierenden Tab, falls möglich")
            print("  none: Öffnet keinen Browser automatisch")
            browser_input = input(f"{Colors.BLUE}Neue Browser-Einstellung (new/existing/none): {Colors.ENDC}")
        
        if browser_input in ["new", "existing", "none"]:
            settings.browser = browser_input
            fancy_print(f"Browser-Einstellung geändert auf: {browser_input}", "success")
        else:
            fancy_print("Ungültige Eingabe. Erlaubte Werte: new, existing, none", "error")
    elif choice == "4":
        # Theme ändern
        if HAS_RICH and console:
            console.print("[bold]Theme-Optionen:[/bold]")
            console.print("  [dim]modern:[/dim] Modernes Design mit Blautönen")
            console.print("  [dim]dark:[/dim] Dunkles Design")
            console.print("  [dim]light:[/dim] Helles Design")
            console.print("  [dim]minimal:[/dim] Minimalistisches Design")
            console.print("  [dim]default:[/dim] Standard-Streamlit-Design")
            theme_input = console.input("[blue]Neues Theme (modern/dark/light/minimal/default): [/blue]")
        else:
            print("Theme-Optionen:")
            print("  modern: Modernes Design mit Blautönen")
            print("  dark: Dunkles Design")
            print("  light: Helles Design")
            print("  minimal: Minimalistisches Design")
            print("  default: Standard-Streamlit-Design")
            theme_input = input(f"{Colors.BLUE}Neues Theme (modern/dark/light/minimal/default): {Colors.ENDC}")
        
        if theme_input in ["modern", "dark", "light", "minimal", "default"]:
            settings.theme = theme_input
            fancy_print(f"Theme geändert auf: {theme_input}", "success")
        else:
            fancy_print("Ungültige Eingabe. Erlaubte Werte: modern, dark, light, minimal, default", "error")
    elif choice == "5":
        # Turbo-Modus ändern
        settings.turbo = not settings.turbo
        fancy_print(f"Turbo-Modus: {'Ein' if settings.turbo else 'Aus'}", "success")
    elif choice == "0":
        # Zurück zum Hauptmenü
        return
    else:
        fancy_print(f"Ungültige Eingabe: {choice}", "error")
    
    # Konfiguration speichern
    settings.save_config()
    
    # Zurück zu den Einstellungen
    configure_settings()


def show_help():
    """Zeigt Hilfe-Informationen zur Anwendung an."""
    if HAS_RICH and console:
        console.print(Panel(
            Text.from_markup(f"""
[bold]Kursplan-Analyse v{APP_VERSION}[/bold]

[bold cyan]Verwendung:[/bold cyan]
  Die Kursplan-Analyse ist eine Anwendung zur Analyse von Stunden- und Kursplänen.
  Sie ermöglicht die Visualisierung und Analyse von Kursüberschneidungen, Stundenplanoptimierung
  und weiteren Auswertungen.

[bold cyan]Hauptfunktionen:[/bold cyan]
  1. [bold]Anwendung starten[/bold] - Startet die Streamlit-basierte Webanwendung
  2. [bold]Umgebung überprüfen[/bold] - Prüft, ob alle Systemvoraussetzungen erfüllt sind
  3. [bold]Abhängigkeiten installieren[/bold] - Installiert erforderliche Python-Pakete
  4. [bold]Fehler beheben[/bold] - Behebt bekannte Probleme in der Anwendung
  5. [bold]Einstellungen anpassen[/bold] - Konfiguriert die Anwendungseinstellungen

[bold cyan]Kommandozeilenoptionen:[/bold cyan]
  --debug       Aktiviert den Debug-Modus
  --port PORT   Gibt den Port an, auf dem die Anwendung laufen soll
  --browser     Steuert das Browser-Verhalten (new/existing/none)
  --theme       Wählt ein Design-Theme (modern/dark/light/minimal/default)
  --turbo       Aktiviert Leistungsoptimierungen
  --check-only  Prüft nur die Umgebung ohne die App zu starten
  --fix         Behebt bekannte Probleme ohne Nachfrage

[bold cyan]Support:[/bold cyan]
  Bei Problemen oder Fragen wenden Sie sich bitte an das Entwicklerteam oder
  lesen Sie die Dokumentation im docs-Verzeichnis.
"""),
            title="Hilfe & Dokumentation",
            border_style="blue",
            padding=(1, 2)
        ))
    else:
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD}{Colors.HEADER}Kursplan-Analyse v{APP_VERSION} - Hilfe & Dokumentation{Colors.ENDC}")
        print("=" * 70)
        print("\nVerwendung:")
        print("  Die Kursplan-Analyse ist eine Anwendung zur Analyse von Stunden- und Kursplänen.")
        print("  Sie ermöglicht die Visualisierung und Analyse von Kursüberschneidungen,")
        print("  Stundenplanoptimierung und weiteren Auswertungen.")
        print("\nHauptfunktionen:")
        print("  1. Anwendung starten - Startet die Streamlit-basierte Webanwendung")
        print("  2. Umgebung überprüfen - Prüft, ob alle Systemvoraussetzungen erfüllt sind")
        print("  3. Abhängigkeiten installieren - Installiert erforderliche Python-Pakete")
        print("  4. Fehler beheben - Behebt bekannte Probleme in der Anwendung")
        print("  5. Einstellungen anpassen - Konfiguriert die Anwendungseinstellungen")
        print("\nKommandozeilenoptionen:")
        print("  --debug       Aktiviert den Debug-Modus")
        print("  --port PORT   Gibt den Port an, auf dem die Anwendung laufen soll")
        print("  --browser     Steuert das Browser-Verhalten (new/existing/none)")
        print("  --theme       Wählt ein Design-Theme (modern/dark/light/minimal/default)")
        print("  --turbo       Aktiviert Leistungsoptimierungen")
        print("  --check-only  Prüft nur die Umgebung ohne die App zu starten")
        print("  --fix         Behebt bekannte Probleme ohne Nachfrage")
        print("\nSupport:")
        print("  Bei Problemen oder Fragen wenden Sie sich bitte an das Entwicklerteam")
        print("  oder lesen Sie die Dokumentation im docs-Verzeichnis.")
        print("\nDrücken Sie Enter, um fortzufahren...", end="")
        input()


if __name__ == "__main__":
    try:
        exit_code = main()
        
        # Menü anzeigen, anstatt das Programm sofort zu beenden
        if HAS_RICH and console:
            show_menu_option = console.input("[bold blue]Möchten Sie zum Hauptmenü wechseln? [Y/n]: [/bold blue]").lower() not in ['n', 'no', 'nein']
        else:
            show_menu_option = input(f"{Colors.BLUE}Möchten Sie zum Hauptmenü wechseln? [Y/n]: {Colors.ENDC}").lower() not in ['n', 'no', 'nein']
        
        if show_menu_option:
            show_menu()
        else:
            # Nach der Ausführung warten, damit das Fenster nicht sofort geschlossen wird
            if platform.system() == "Windows":
                print("\nDrücken Sie eine beliebige Taste, um das Programm zu beenden...")
                os.system("pause >nul")
            else:
                input("\nDrücken Sie Enter, um das Programm zu beenden...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        fancy_print("\nProgramm durch Benutzer abgebrochen", "warning")
        # Warten auf Benutzereingabe vor dem Schließen
        if platform.system() == "Windows":
            os.system("pause >nul")
        else:
            input("Drücken Sie Enter, um fortzufahren...")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unerwarteter Fehler beim Starten der Anwendung")
        fancy_print(f"Unerwarteter Fehler: {e}", "error")
        # Warten auf Benutzereingabe vor dem Schließen
        if platform.system() == "Windows":
            os.system("pause >nul")
        else:
            input("Drücken Sie Enter, um fortzufahren...")
        sys.exit(1)
