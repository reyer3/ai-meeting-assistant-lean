#!/usr/bin/env python3
"""
Script de instalación lean para AI Meeting Assistant
Automatiza todo el setup inicial
"""

import os
import sys
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

console = Console()

class LeanInstaller:
    """Instalador automático para configuración lean"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.data_dir = self.project_root / "data"
        
        self.steps_completed = []
        self.requirements_installed = False
    
    def show_intro(self):
        """Muestra introducción del instalador"""
        console.print(Panel.fit(
            "[bold blue]🚀 AI Meeting Assistant Lean - Instalador[/bold blue]\\n\\n"
            "Este instalador configurará automáticamente:\\n"
            "• Dependencias de Python\\n"
            "• Modelos de IA (Ollama + Whisper)\\n"
            "• Base de conocimiento AEIOU\\n"
            "• Perfil de voz personal\\n\\n"
            "[yellow]Tiempo estimado: 10-15 minutos[/yellow]\\n"
            "[dim]Requiere conexión a internet para descargar modelos[/dim]",
            border_style="blue"
        ))
    
    def check_system_requirements(self) -> bool:
        """Verifica requerimientos del sistema"""
        console.print("\\n[cyan]🔍 Verificando requerimientos del sistema...[/cyan]")
        
        # Verificar Python
        python_version = sys.version_info
        if python_version < (3, 8):
            console.print(f"[red]❌ Python {python_version.major}.{python_version.minor} detectado. Se requiere Python 3.8+[/red]")
            return False
        
        console.print(f"[green]✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}[/green]")
        
        # Verificar pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            console.print("[green]✅ pip disponible[/green]")
        except subprocess.CalledProcessError:
            console.print("[red]❌ pip no disponible[/red]")
            return False
        
        # Verificar espacio en disco (mínimo 3GB)
        try:
            import shutil
            free_space = shutil.disk_usage(self.project_root).free / (1024**3)  # GB
            if free_space < 3:
                console.print(f"[yellow]⚠️ Espacio disponible: {free_space:.1f}GB (recomendado: 3GB+)[/yellow]")
            else:
                console.print(f"[green]✅ Espacio disponible: {free_space:.1f}GB[/green]")
        except:
            console.print("[yellow]⚠️ No se pudo verificar espacio en disco[/yellow]")
        
        return True
    
    def install_python_dependencies(self) -> bool:
        """Instala dependencias de Python"""
        console.print("\\n[cyan]📦 Instalando dependencias de Python...[/cyan]")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            console.print("[red]❌ requirements.txt no encontrado[/red]")
            return False
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Instalando dependencias...", total=None)
                
                # Instalar requirements
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    console.print(f"[red]❌ Error instalando dependencias:[/red]")
                    console.print(result.stderr)
                    return False
                
                progress.update(task, description="✅ Dependencias instaladas")
            
            console.print("[green]✅ Dependencias de Python instaladas[/green]")
            self.requirements_installed = True
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False
    
    def install_ollama(self) -> bool:
        """Instala Ollama y el modelo qwen2.5:0.5b"""
        console.print("\\n[cyan]🧠 Configurando Ollama...[/cyan]")
        
        # Verificar si Ollama ya está instalado
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                console.print("[green]✅ Ollama ya está instalado[/green]")
            else:
                raise subprocess.CalledProcessError(1, "ollama")
                
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            console.print("[yellow]⚠️ Ollama no encontrado[/yellow]")
            console.print("\\n[cyan]Para instalar Ollama:[/cyan]")
            console.print("  Windows/Mac: https://ollama.ai/download")
            console.print("  Linux: curl -fsSL https://ollama.ai/install.sh | sh")
            
            if not Confirm.ask("¿Has instalado Ollama y quieres continuar?"):
                return False
        
        # Verificar/instalar modelo qwen2.5:0.5b
        console.print("\\n[cyan]📥 Verificando modelo qwen2.5:0.5b...[/cyan]")
        
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            
            if "qwen2.5:0.5b" in result.stdout:
                console.print("[green]✅ Modelo qwen2.5:0.5b ya está disponible[/green]")
                return True
            
            # Descargar modelo
            console.print("[yellow]📥 Descargando modelo qwen2.5:0.5b (~316MB)...[/yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                task = progress.add_task("Descargando modelo...", total=None)
                
                result = subprocess.run(
                    ["ollama", "pull", "qwen2.5:0.5b"],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutos max
                )
                
                if result.returncode != 0:
                    console.print(f"[red]❌ Error descargando modelo:[/red]")
                    console.print(result.stderr)
                    return False
                
                progress.update(task, description="✅ Modelo descargado")
            
            console.print("[green]✅ Modelo qwen2.5:0.5b instalado[/green]")
            return True
            
        except subprocess.TimeoutExpired:
            console.print("[red]❌ Timeout descargando modelo[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False
    
    def setup_knowledge_base(self) -> bool:
        """Configura la base de conocimiento"""
        console.print("\\n[cyan]📚 Configurando base de conocimiento...[/cyan]")
        
        setup_script = self.scripts_dir / "setup_knowledge_base.py"
        
        if not setup_script.exists():
            console.print(f"[red]❌ Script no encontrado: {setup_script}[/red]")
            return False
        
        try:
            # Ejecutar setup automáticamente (con respuesta 'y')
            process = subprocess.Popen(
                [sys.executable, str(setup_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input="y\\n", timeout=120)
            
            if process.returncode == 0:
                console.print("[green]✅ Base de conocimiento configurada[/green]")
                return True
            else:
                console.print(f"[red]❌ Error configurando knowledge base:[/red]")
                console.print(stderr)
                return False
                
        except subprocess.TimeoutExpired:
            console.print("[red]❌ Timeout configurando knowledge base[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False
    
    def setup_voice_profile(self) -> bool:
        """Configura el perfil de voz (interactivo)"""
        console.print("\\n[cyan]🎤 Configurando perfil de voz...[/cyan]")
        
        voice_profile_path = self.data_dir / "user_voice_profile.pkl"
        
        if voice_profile_path.exists():
            if not Confirm.ask("Ya existe un perfil de voz. ¿Recrear?", default=False):
                console.print("[green]✅ Usando perfil de voz existente[/green]")
                return True
        
        setup_script = self.scripts_dir / "setup_voice_profile.py"
        
        if not setup_script.exists():
            console.print(f"[red]❌ Script no encontrado: {setup_script}[/red]")
            return False
        
        console.print("\\n[yellow]🎙️ Configuración interactiva del perfil de voz[/yellow]")
        console.print("[dim]Se abrirá el setup interactivo. Sigue las instrucciones...[/dim]")
        
        if not Confirm.ask("¿Continuar con setup de voz?"):
            console.print("[yellow]⏭️ Saltando setup de voz[/yellow]")
            return True
        
        try:
            # Ejecutar setup interactivo
            result = subprocess.run([sys.executable, str(setup_script)], 
                                  timeout=600)  # 10 minutos max
            
            if result.returncode == 0:
                console.print("[green]✅ Perfil de voz configurado[/green]")
                return True
            else:
                console.print("[yellow]⚠️ Setup de voz cancelado o falló[/yellow]")
                return True  # Continuar sin perfil por ahora
                
        except subprocess.TimeoutExpired:
            console.print("[red]❌ Timeout en setup de voz[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False
    
    def create_startup_scripts(self):
        """Crea scripts de inicio convenientes"""
        console.print("\\n[cyan]📝 Creando scripts de inicio...[/cyan]")
        
        # Script para Windows
        bat_content = '''@echo off
echo 🚀 Iniciando AI Meeting Assistant Lean...
cd /d "%~dp0"
python src/main.py
pause
'''
        
        with open(self.project_root / "start_lean.bat", 'w') as f:
            f.write(bat_content)
        
        # Script para Linux/Mac
        sh_content = '''#!/bin/bash
echo "🚀 Iniciando AI Meeting Assistant Lean..."
cd "$(dirname "$0")"
python src/main.py
'''
        
        sh_file = self.project_root / "start_lean.sh"
        with open(sh_file, 'w') as f:
            f.write(sh_content)
        
        # Hacer ejecutable en Linux/Mac
        try:
            os.chmod(sh_file, 0o755)
        except:
            pass
        
        console.print("[green]✅ Scripts de inicio creados[/green]")
    
    def run_installation(self) -> bool:
        """Ejecuta la instalación completa"""
        
        self.show_intro()
        
        if not Confirm.ask("\\n¿Continuar con la instalación?"):
            console.print("[yellow]Instalación cancelada[/yellow]")
            return False
        
        # Verificar requerimientos
        if not self.check_system_requirements():
            console.print("\\n[red]❌ Requerimientos del sistema no cumplidos[/red]")
            return False
        
        # Crear directorios
        self.data_dir.mkdir(exist_ok=True)
        
        # Pasos de instalación
        steps = [
            ("Dependencias Python", self.install_python_dependencies),
            ("Ollama y modelo IA", self.install_ollama),
            ("Base de conocimiento", self.setup_knowledge_base),
            ("Perfil de voz", self.setup_voice_profile),
            ("Scripts de inicio", lambda: (self.create_startup_scripts(), True)[1])
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            console.print(f"\\n{'='*60}")
            console.print(f"[bold cyan]📋 Paso: {step_name}[/bold cyan]")
            console.print("="*60)
            
            try:
                if step_func():
                    self.steps_completed.append(step_name)
                    console.print(f"[green]✅ {step_name} completado[/green]")
                else:
                    failed_steps.append(step_name)
                    console.print(f"[red]❌ {step_name} falló[/red]")
                    
                    # Preguntar si continuar
                    if step_name != "Perfil de voz":  # El perfil de voz es opcional
                        if not Confirm.ask(f"¿Continuar sin {step_name}?", default=False):
                            break
            
            except Exception as e:
                console.print(f"[red]❌ Error en {step_name}: {e}[/red]")
                failed_steps.append(step_name)
                
                if not Confirm.ask("¿Continuar con la instalación?", default=False):
                    break
        
        # Mostrar resumen
        self.show_installation_summary(failed_steps)
        
        return len(failed_steps) == 0
    
    def show_installation_summary(self, failed_steps: list):
        """Muestra resumen de la instalación"""
        
        console.print("\\n" + "="*80)
        console.print("[bold blue]📊 RESUMEN DE INSTALACIÓN[/bold blue]")
        console.print("="*80)
        
        console.print(f"\\n[green]✅ Pasos completados: {len(self.steps_completed)}[/green]")
        for step in self.steps_completed:
            console.print(f"  ✓ {step}")
        
        if failed_steps:
            console.print(f"\\n[red]❌ Pasos fallidos: {len(failed_steps)}[/red]")
            for step in failed_steps:
                console.print(f"  ✗ {step}")
        
        console.print("\\n[bold cyan]🚀 CÓMO INICIAR EL ASISTENTE:[/bold cyan]")
        
        if len(failed_steps) == 0:
            console.print("\\n[green]Todo listo! Puedes iniciar de varias formas:[/green]")
            console.print("  1. [bold]python src/main.py[/bold]")
            console.print("  2. [bold]./start_lean.sh[/bold] (Linux/Mac)")
            console.print("  3. [bold]start_lean.bat[/bold] (Windows)")
        else:
            console.print("\\n[yellow]⚠️ Algunos pasos fallaron. Para completar manualmente:[/yellow]")
            
            if "Ollama y modelo IA" in failed_steps:
                console.print("  • Instalar Ollama: https://ollama.ai")
                console.print("  • Ejecutar: ollama pull qwen2.5:0.5b")
            
            if "Base de conocimiento" in failed_steps:
                console.print("  • Ejecutar: python scripts/setup_knowledge_base.py")
            
            if "Perfil de voz" in failed_steps:
                console.print("  • Ejecutar: python scripts/setup_voice_profile.py")
        
        console.print("\\n[dim]Para más información, consulta el README.md[/dim]")


def main():
    """Función principal"""
    installer = LeanInstaller()
    
    try:
        success = installer.run_installation()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        console.print("\\n[yellow]⏹️ Instalación interrumpida por el usuario[/yellow]")
        sys.exit(1)
        
    except Exception as e:
        console.print(f"\\n[red]❌ Error inesperado: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
