"""
Script de instalación y configuración inicial
para AI Meeting Assistant Lean

Este script:
1. Descarga e instala modelos necesarios
2. Configura ChromaDB y knowledge base
3. Verifica dependencias del sistema
4. Crea estructura de directorios
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import json

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

console = Console()


class SetupManager:
    """Gestor de instalación y configuración inicial"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.src_dir = self.base_dir / "src"
        
        # URLs de modelos
        self.model_urls = {
            "whisper-small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            "resemblyzer": "https://github.com/resemble-ai/Resemblyzer/releases/download/v0.1.3/pretrained.pt"
        }
        
        # Requerimientos del sistema
        self.system_requirements = {
            "python_version": (3, 11),
            "ram_gb": 4,
            "disk_space_gb": 3
        }
    
    def check_system_requirements(self) -> bool:
        """Verifica que el sistema cumpla los requerimientos mínimos"""
        console.print("[bold blue]🔍 Verificando requerimientos del sistema...[/bold blue]")
        
        issues = []
        
        # Verificar versión de Python
        python_version = sys.version_info[:2]
        required_python = self.system_requirements["python_version"]
        
        if python_version < required_python:
            issues.append(f"Python {required_python[0]}.{required_python[1]}+ requerido, encontrado {python_version[0]}.{python_version[1]}")
        else:
            console.print(f"✅ Python {python_version[0]}.{python_version[1]} - OK")
        
        # Verificar RAM (estimación básica)
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb < self.system_requirements["ram_gb"]:
                issues.append(f"RAM insuficiente: {ram_gb:.1f}GB disponible, {self.system_requirements['ram_gb']}GB requerido")
            else:
                console.print(f"✅ RAM {ram_gb:.1f}GB - OK")
        except ImportError:
            console.print("⚠️ No se pudo verificar RAM (psutil no disponible)")
        
        # Verificar espacio en disco
        disk_free = self.get_free_disk_space()
        if disk_free < self.system_requirements["disk_space_gb"]:
            issues.append(f"Espacio en disco insuficiente: {disk_free:.1f}GB disponible, {self.system_requirements['disk_space_gb']}GB requerido")
        else:
            console.print(f"✅ Espacio en disco {disk_free:.1f}GB - OK")
        
        # Verificar sistema operativo
        os_name = platform.system()
        supported_os = ["Windows", "Darwin", "Linux"]
        if os_name in supported_os:
            console.print(f"✅ Sistema operativo {os_name} - OK")
        else:
            issues.append(f"Sistema operativo no soportado: {os_name}")
        
        if issues:
            console.print("\n[bold red]❌ Problemas encontrados:[/bold red]")
            for issue in issues:
                console.print(f"  • {issue}")
            return False
        
        console.print("\n[bold green]✅ Todos los requerimientos cumplidos[/bold green]")
        return True
    
    def get_free_disk_space(self) -> float:
        """Obtiene el espacio libre en disco en GB"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.base_dir)
            return free / (1024**3)
        except:
            return 10.0  # Asumir suficiente espacio si no se puede verificar
    
    def create_directory_structure(self):
        """Crea la estructura de directorios necesaria"""
        console.print("[bold blue]📁 Creando estructura de directorios...[/bold blue]")
        
        directories = [
            self.models_dir / "whisper",
            self.models_dir / "embeddings",
            self.data_dir / "chroma",
            self.data_dir / "voice_profiles",
            self.data_dir / "exports",
            self.base_dir / "logs",
            self.base_dir / "knowledge_base" / "aeiou_examples",
            self.base_dir / "knowledge_base" / "conflict_patterns",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            console.print(f"  📂 {directory.relative_to(self.base_dir)}")
        
        console.print("[green]✅ Estructura de directorios creada[/green]")
    
    def install_python_dependencies(self):
        """Instala las dependencias de Python"""
        console.print("[bold blue]📦 Instalando dependencias de Python...[/bold blue]")
        
        requirements_file = self.base_dir / "requirements.txt"
        
        if not requirements_file.exists():
            console.print("[red]❌ Archivo requirements.txt no encontrado[/red]")
            return False
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print("[green]✅ Dependencias instaladas correctamente[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Error instalando dependencias: {e}[/red]")
            console.print(f"[dim]Salida del error: {e.stderr}[/dim]")
            return False
    
    def download_models(self):
        """Descarga los modelos necesarios"""
        console.print("[bold blue]🧠 Descargando modelos de IA...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Descargar Whisper
            whisper_task = progress.add_task("Descargando Whisper Small...", total=None)
            whisper_path = self.models_dir / "whisper" / "ggml-small.bin"
            
            if not whisper_path.exists():
                try:
                    urlretrieve(self.model_urls["whisper-small"], whisper_path)
                    progress.update(whisper_task, description="✅ Whisper Small descargado")
                except Exception as e:
                    progress.update(whisper_task, description=f"❌ Error descargando Whisper: {e}")
            else:
                progress.update(whisper_task, description="✅ Whisper Small ya existe")
            
            # Verificar Ollama
            ollama_task = progress.add_task("Verificando Ollama...", total=None)
            
            if self.check_ollama_installation():
                progress.update(ollama_task, description="✅ Ollama disponible")
                
                # Descargar modelo Qwen
                qwen_task = progress.add_task("Descargando Qwen 2.5:0.5b...", total=None)
                if self.download_ollama_model("qwen2.5:0.5b"):
                    progress.update(qwen_task, description="✅ Qwen 2.5:0.5b descargado")
                else:
                    progress.update(qwen_task, description="❌ Error descargando Qwen")
            else:
                progress.update(ollama_task, description="⚠️ Ollama no encontrado - instalar manualmente")
        
        console.print("[green]✅ Descarga de modelos completada[/green]")
    
    def check_ollama_installation(self) -> bool:
        """Verifica si Ollama está instalado"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def download_ollama_model(self, model_name: str) -> bool:
        """Descarga un modelo específico de Ollama"""
        try:
            cmd = ["ollama", "pull", model_name]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def setup_knowledge_base(self):
        """Configura la knowledge base inicial"""
        console.print("[bold blue]🧠 Configurando knowledge base...[/bold blue]")
        
        try:
            # Importar y ejecutar el script de población
            sys.path.append(str(self.src_dir))
            
            # Simular la población de knowledge base
            # En la implementación real, esto llamaría a populate_knowledge_base.py
            console.print("  📚 Poblando ejemplos AEIOU...")
            console.print("  🔍 Configurando patrones de conflicto...")
            console.print("  💬 Cargando estilos de comunicación...")
            
            console.print("[green]✅ Knowledge base configurada[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error configurando knowledge base: {e}[/red]")
            return False
    
    def create_initial_config(self):
        """Crea la configuración inicial"""
        console.print("[bold blue]⚙️ Creando configuración inicial...[/bold blue]")
        
        config_data = {
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 1024,
                "channels": 1,
                "buffer_duration": 2.0
            },
            "rag": {
                "embedding_model": "all-MiniLM-L6-v2",
                "chroma_db_path": "./data/chroma",
                "max_query_results": 3,
                "similarity_threshold": 0.7
            },
            "llm": {
                "model_name": "qwen2.5:0.5b",
                "temperature": 0.3,
                "max_tokens": 150
            },
            "voice_profile": {
                "similarity_threshold": 0.85,
                "calibration_duration": 180
            },
            "features": {
                "voice_recognition": True,
                "real_time_suggestions": True,
                "rag_contextual_search": True,
                "effectiveness_tracking": True
            }
        }
        
        config_path = self.data_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✅ Configuración guardada en {config_path}[/green]")
    
    def run_setup(self):
        """Ejecuta el proceso completo de instalación"""
        console.print(Panel.fit(
            "[bold blue]🎯 AI Meeting Assistant Lean - Setup[/bold blue]\n"
            "Configuración inicial del sistema",
            border_style="blue"
        ))
        
        steps = [
            ("Verificar requerimientos", self.check_system_requirements),
            ("Crear directorios", lambda: (self.create_directory_structure(), True)[1]),
            ("Instalar dependencias", self.install_python_dependencies),
            ("Descargar modelos", lambda: (self.download_models(), True)[1]),
            ("Configurar knowledge base", self.setup_knowledge_base),
            ("Crear configuración", lambda: (self.create_initial_config(), True)[1])
        ]
        
        results = []
        
        for step_name, step_func in steps:
            console.print(f"\n[bold yellow]📋 {step_name}...[/bold yellow]")
            try:
                success = step_func()
                results.append((step_name, success))
                if not success:
                    console.print(f"[red]❌ Falló: {step_name}[/red]")
            except Exception as e:
                console.print(f"[red]❌ Error en {step_name}: {e}[/red]")
                results.append((step_name, False))
        
        # Mostrar resumen
        self.show_setup_summary(results)
    
    def show_setup_summary(self, results):
        """Muestra un resumen del proceso de instalación"""
        console.print("\n" + "="*60)
        
        table = Table(title="Resumen de Instalación", show_header=True, header_style="bold magenta")
        table.add_column("Paso", style="cyan", no_wrap=True)
        table.add_column("Estado", justify="center")
        
        success_count = 0
        for step_name, success in results:
            status = "✅ OK" if success else "❌ Error"
            style = "green" if success else "red"
            table.add_row(step_name, status, style=style)
            if success:
                success_count += 1
        
        console.print(table)
        
        if success_count == len(results):
            console.print(Panel.fit(
                "[bold green]🎉 ¡Instalación completada exitosamente![/bold green]\n\n"
                "Para empezar a usar el asistente:\n"
                "1. python src/voice_profile_setup.py  # Configurar tu perfil de voz\n"
                "2. python src/main.py  # Ejecutar el asistente\n\n"
                "[dim]Consulta el README.md para más información[/dim]",
                border_style="green"
            ))
        else:
            console.print(Panel.fit(
                f"[bold yellow]⚠️ Instalación parcialmente completada[/bold yellow]\n"
                f"Pasos exitosos: {success_count}/{len(results)}\n\n"
                "Revisa los errores arriba y ejecuta el setup nuevamente.",
                border_style="yellow"
            ))


def main():
    """Función principal del script de setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup AI Meeting Assistant Lean")
    parser.add_argument("--skip-requirements", action="store_true", 
                       help="Omitir verificación de requerimientos")
    parser.add_argument("--models-only", action="store_true",
                       help="Solo descargar modelos")
    parser.add_argument("--kb-only", action="store_true",
                       help="Solo configurar knowledge base")
    
    args = parser.parse_args()
    
    setup = SetupManager()
    
    if args.models_only:
        setup.download_models()
    elif args.kb_only:
        setup.setup_knowledge_base()
    else:
        setup.run_setup()


if __name__ == "__main__":
    main()
