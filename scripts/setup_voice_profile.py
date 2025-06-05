#!/usr/bin/env python3
"""
Setup de perfil de voz personal para identificación lean
Crea embeddings de voz para diferenciar tu voz de otros participantes
"""

import asyncio
import time
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from loguru import logger

# Imports del sistema
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from audio.system_capture import SystemAudioCapture

console = Console()

class VoiceProfileSetup:
    """Setup de perfil de voz personal"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.samples_needed = 5  # 5 muestras de voz
        self.sample_duration = 5  # 5 segundos cada una
        
        self.voice_samples: List[np.ndarray] = []
        self.profile_data = {
            "samples": [],
            "embeddings": [],
            "sample_rate": self.sample_rate,
            "created_at": time.time(),
            "version": "1.0"
        }
    
    def show_intro(self):
        """Muestra introducción del setup"""
        console.print(Panel.fit(
            "[bold blue]🎤 Setup de Perfil de Voz Personal[/bold blue]\n\n"
            "Este asistente creará tu perfil de voz para:\n"
            "• Identificarte automáticamente en reuniones\n"
            "• Diferenciar tu voz de otros participantes\n"
            "• Generar sugerencias solo cuando hablas tú\n\n"
            "[dim]El proceso toma ~5 minutos y solo se hace una vez[/dim]",
            border_style="blue"
        ))
    
    def check_audio_setup(self) -> bool:
        """Verifica configuración de audio"""
        console.print("\n[cyan]📱 Verificando dispositivos de audio...[/cyan]")
        
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if not input_devices:
                console.print("[red]❌ No se encontraron dispositivos de entrada[/red]")
                return False
            
            console.print(f"[green]✅ Encontrados {len(input_devices)} dispositivos de entrada[/green]")
            
            # Mostrar dispositivos disponibles
            console.print("\n[dim]Dispositivos disponibles:[/dim]")
            for i, device in enumerate(input_devices):
                console.print(f"  {i+1}: {device['name']}")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error verificando audio: {e}[/red]")
            return False
    
    def record_voice_sample(self, sample_num: int) -> Optional[np.ndarray]:
        """Graba una muestra de voz"""
        
        console.print(f"\\n[bold yellow]🎙️ Grabando muestra {sample_num}/{self.samples_needed}[/bold yellow]")
        
        # Frases sugeridas para mayor variabilidad
        phrases = [
            "Hola, soy [tu nombre] y estoy configurando mi perfil de voz para el asistente de IA.",
            "Esta es mi voz natural hablando en una reunión de trabajo típica.",
            "Me gusta participar en discusiones técnicas y dar mi opinión sobre diferentes temas.",
            "Cuando hablo en reuniones, normalmente uso un tono profesional pero relajado.",
            "Este sistema me ayudará a recibir sugerencias inteligentes durante mis reuniones."
        ]
        
        suggested_phrase = phrases[sample_num - 1]
        
        console.print(f"\n[cyan]💬 Frase sugerida:[/cyan]")
        console.print(f'[italic]"{suggested_phrase}"[/italic]')
        console.print("[dim]Puedes usar esta frase o hablar libremente por 5 segundos[/dim]")
        
        # Countdown
        for i in range(3, 0, -1):
            console.print(f"[yellow]Empezando en {i}...[/yellow]")
            time.sleep(1)
        
        console.print("[bold green]🔴 GRABANDO... (5 segundos)[/bold green]")
        
        try:
            # Grabar audio
            audio_data = sd.rec(
                int(self.sample_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Esperar a que termine
            
            # Verificar calidad de la grabación
            rms_level = np.sqrt(np.mean(audio_data**2))
            
            if rms_level < 0.01:
                console.print("[yellow]⚠️ Audio muy bajo. ¿Quieres repetir esta muestra?[/yellow]")
                if Prompt.ask("Repetir grabación", choices=["y", "n"], default="y") == "y":
                    return self.record_voice_sample(sample_num)
            
            console.print(f"[green]✅ Muestra grabada (nivel: {rms_level:.3f})[/green]")
            
            # Reproducir para confirmar
            if Prompt.ask("¿Quieres escuchar la grabación?", choices=["y", "n"], default="n") == "y":
                console.print("[blue]🔊 Reproduciendo...[/blue]")
                sd.play(audio_data, self.sample_rate)
                sd.wait()
            
            # Confirmar calidad
            if Prompt.ask("¿La grabación suena bien?", choices=["y", "n"], default="y") == "y":
                return audio_data.flatten()
            else:
                return self.record_voice_sample(sample_num)
                
        except Exception as e:
            console.print(f"[red]❌ Error grabando: {e}[/red]")
            return None
    
    def process_voice_samples(self):
        """Procesa las muestras de voz y crea embeddings"""
        console.print("\\n[cyan]🧠 Procesando muestras de voz...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Creando embeddings de voz...", total=len(self.voice_samples))
            
            for i, sample in enumerate(self.voice_samples):
                # Para desarrollo lean, usar características simples pero efectivas
                embedding = self._create_simple_voice_embedding(sample)
                
                self.profile_data["samples"].append(sample.tolist())
                self.profile_data["embeddings"].append(embedding.tolist())
                
                progress.update(task, advance=1)
                time.sleep(0.5)  # Simular procesamiento
        
        console.print("[green]✅ Embeddings creados exitosamente[/green]")
    
    def _create_simple_voice_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Crea embedding simple pero efectivo de voz"""
        
        # Características básicas pero distintivas
        features = []
        
        # 1. Características espectrales
        fft = np.fft.fft(audio)
        magnitude_spectrum = np.abs(fft[:len(fft)//2])
        
        # Energía en diferentes bandas de frecuencia
        bands = np.array_split(magnitude_spectrum, 8)
        band_energies = [np.sum(band) for band in bands]
        features.extend(band_energies)
        
        # 2. Características temporales
        features.append(np.mean(audio))  # DC component
        features.append(np.std(audio))   # Variabilidad
        features.append(np.max(audio))   # Peak level
        
        # 3. Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        features.append(zero_crossings / len(audio))
        
        # 4. Características de pitch (fundamental frequency)
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Encontrar picos en autocorrelación
        if len(autocorr) > 50:
            peak_idx = np.argmax(autocorr[20:200]) + 20  # Evitar DC
            fundamental_freq = self.sample_rate / peak_idx if peak_idx > 0 else 0
            features.append(fundamental_freq)
        else:
            features.append(0)
        
        # Normalizar características
        features = np.array(features, dtype=np.float32)
        features = features / (np.linalg.norm(features) + 1e-8)  # Evitar división por cero
        
        return features
    
    def save_profile(self) -> bool:
        """Guarda el perfil de voz"""
        
        # Crear directorio data si no existe
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        profile_path = data_dir / "user_voice_profile.pkl"
        
        try:
            with open(profile_path, 'wb') as f:
                pickle.dump(self.profile_data, f)
            
            console.print(f"[green]✅ Perfil guardado en: {profile_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error guardando perfil: {e}[/red]")
            return False
    
    def run_setup(self) -> bool:
        """Ejecuta el setup completo"""
        
        self.show_intro()
        
        # Verificar audio
        if not self.check_audio_setup():
            console.print("\\n[red]No se puede continuar sin dispositivos de audio[/red]")
            return False
        
        # Confirmar inicio
        if Prompt.ask("\\n¿Continuar con la grabación?", choices=["y", "n"], default="y") != "y":
            console.print("[yellow]Setup cancelado[/yellow]")
            return False
        
        # Grabar muestras
        console.print(f"\\n[bold cyan]📼 Grabando {self.samples_needed} muestras de voz[/bold cyan]")
        
        for i in range(1, self.samples_needed + 1):
            sample = self.record_voice_sample(i)
            
            if sample is None:
                console.print("[red]❌ Error grabando muestra. Cancelando setup.[/red]")
                return False
            
            self.voice_samples.append(sample)
            
            if i < self.samples_needed:
                console.print("[dim]Toma un respiro de 3 segundos...[/dim]")
                time.sleep(3)
        
        # Procesar muestras
        self.process_voice_samples()
        
        # Guardar perfil
        if not self.save_profile():
            return False
        
        # Mostrar resumen
        console.print("\\n" + "="*60)
        console.print("[bold green]🎉 ¡Perfil de voz creado exitosamente![/bold green]")
        console.print("="*60)
        console.print(f"📊 Muestras procesadas: {len(self.voice_samples)}")
        console.print(f"🧠 Embeddings creados: {len(self.profile_data['embeddings'])}")
        console.print(f"📁 Guardado en: data/user_voice_profile.pkl")
        console.print("\\n[cyan]🚀 Ya puedes ejecutar el asistente principal:[/cyan]")
        console.print("[bold]python src/main.py[/bold]")
        
        return True


def main():
    """Función principal"""
    
    # Configurar logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
              format="<green>{time:HH:mm:ss}</green> | {message}")
    
    setup = VoiceProfileSetup()
    
    try:
        success = setup.run_setup()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        console.print("\\n[yellow]⏹️ Setup interrumpido por el usuario[/yellow]")
        exit_code = 1
        
    except Exception as e:
        console.print(f"\\n[red]❌ Error inesperado: {e}[/red]")
        logger.exception("Error en setup de voz")
        exit_code = 1
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
