#!/usr/bin/env python3
"""
Overlay UI simple y no intrusivo para mostrar sugerencias AEIOU
Enfoque lean: máxima funcionalidad con mínima complejidad
"""

import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from loguru import logger

@dataclass
class OverlayConfig:
    """Configuración del overlay"""
    
    # Posición y tamaño
    width: int = 400
    height: int = 150
    x_offset: int = 50  # Desde la derecha
    y_offset: int = 100  # Desde arriba
    
    # Comportamiento
    always_on_top: bool = True
    auto_hide_delay: float = 10.0  # Ocultar después de 10s
    fade_duration: float = 0.3  # Animación de fade
    
    # Styling
    background_color: str = "#2D3748"  # Gris oscuro
    text_color: str = "#E2E8F0"  # Gris claro
    border_color: str = "#4A5568"  # Gris medio
    suggestion_color: str = "#68D391"  # Verde suave
    
    # Fonts
    title_font: tuple = ("Segoe UI", 10, "bold")
    text_font: tuple = ("Segoe UI", 9)
    small_font: tuple = ("Segoe UI", 8)


class SuggestionOverlay:
    """
    Overlay no intrusivo para mostrar sugerencias AEIOU
    """
    
    def __init__(self, config: OverlayConfig = None):
        self.config = config or OverlayConfig()
        self.root: Optional[tk.Tk] = None
        self.is_visible = False
        self.auto_hide_timer: Optional[threading.Timer] = None
        
        # Referencias a widgets
        self.title_label: Optional[tk.Label] = None
        self.suggestion_text: Optional[tk.Text] = None
        self.info_label: Optional[tk.Label] = None
        self.close_button: Optional[tk.Button] = None
        
        # Callback para cuando usuario cierra sugerencia
        self.on_close_callback: Optional[Callable] = None
        
        logger.info("🖼️ Suggestion Overlay inicializado")
    
    def initialize(self):
        """Inicializa la ventana del overlay"""
        try:
            self.root = tk.Tk()
            self.root.title("AI Meeting Assistant")
            
            # Configuración de la ventana
            self._setup_window()
            self._create_widgets()
            self._setup_bindings()
            
            # Inicialmente oculto
            self.hide()
            
            logger.success("✅ Overlay UI inicializado")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando overlay: {e}")
            raise
    
    def _setup_window(self):
        """Configura propiedades de la ventana"""
        
        # Tamaño y posición
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        x = screen_width - self.config.width - self.config.x_offset
        y = self.config.y_offset
        
        self.root.geometry(f"{self.config.width}x{self.config.height}+{x}+{y}")
        
        # Propiedades de ventana
        self.root.configure(bg=self.config.background_color)
        self.root.resizable(False, False)
        
        # Always on top
        if self.config.always_on_top:
            self.root.attributes('-topmost', True)
        
        # Sin decoraciones de ventana (más limpio)
        self.root.overrideredirect(True)
        
        # Transparencia parcial
        self.root.attributes('-alpha', 0.95)
    
    def _create_widgets(self):
        """Crea los widgets del overlay"""
        
        # Frame principal con borde
        main_frame = tk.Frame(
            self.root,
            bg=self.config.background_color,
            highlightbackground=self.config.border_color,
            highlightthickness=2
        )
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Header con título y botón cerrar
        header_frame = tk.Frame(main_frame, bg=self.config.background_color)
        header_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        self.title_label = tk.Label(
            header_frame,
            text="💡 Sugerencia AEIOU",
            font=self.config.title_font,
            fg=self.config.suggestion_color,
            bg=self.config.background_color
        )
        self.title_label.pack(side=tk.LEFT)
        
        self.close_button = tk.Button(
            header_frame,
            text="✕",
            font=("Arial", 8),
            fg=self.config.text_color,
            bg=self.config.background_color,
            bd=0,
            highlightthickness=0,
            command=self._on_close_clicked
        )
        self.close_button.pack(side=tk.RIGHT)
        
        # Área de texto para la sugerencia
        text_frame = tk.Frame(main_frame, bg=self.config.background_color)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        self.suggestion_text = tk.Text(
            text_frame,
            height=4,
            font=self.config.text_font,
            fg=self.config.text_color,
            bg=self.config.background_color,
            bd=0,
            highlightthickness=0,
            wrap=tk.WORD,
            state=tk.DISABLED,
            cursor="arrow"
        )
        self.suggestion_text.pack(fill=tk.BOTH, expand=True)
        
        # Footer con información
        footer_frame = tk.Frame(main_frame, bg=self.config.background_color)
        footer_frame.pack(fill=tk.X, padx=8, pady=(4, 8))
        
        self.info_label = tk.Label(
            footer_frame,
            text="",
            font=self.config.small_font,
            fg=self.config.text_color,
            bg=self.config.background_color
        )
        self.info_label.pack(side=tk.LEFT)
        
        # Label de tiempo restante
        self.timer_label = tk.Label(
            footer_frame,
            text="",
            font=self.config.small_font,
            fg=self.config.text_color,
            bg=self.config.background_color
        )
        self.timer_label.pack(side=tk.RIGHT)
    
    def _setup_bindings(self):
        """Configura event bindings"""
        
        # Click para cerrar
        self.root.bind("<Button-1>", self._on_click)
        
        # Escape para cerrar
        self.root.bind("<Escape>", lambda e: self._on_close_clicked())
        
        # Hover effects
        self.close_button.bind("<Enter>", self._on_close_hover)
        self.close_button.bind("<Leave>", self._on_close_leave)
    
    def show_suggestion(self, 
                       suggestion_text: str,
                       context_info: str = "",
                       confidence: float = 0.8):
        """
        Muestra una sugerencia en el overlay
        """
        try:
            if not self.root:
                logger.warning("⚠️ Overlay no inicializado")
                return
            
            # Actualizar contenido
            self.suggestion_text.config(state=tk.NORMAL)
            self.suggestion_text.delete(1.0, tk.END)
            self.suggestion_text.insert(1.0, suggestion_text)
            self.suggestion_text.config(state=tk.DISABLED)
            
            # Actualizar info
            info_text = f"Confianza: {confidence:.0%}"
            if context_info:
                info_text += f" | {context_info}"
            
            self.info_label.config(text=info_text)
            
            # Mostrar overlay
            self.show()
            
            # Configurar auto-hide
            self._start_auto_hide_timer()
            
            logger.info(f"💡 Sugerencia mostrada: {suggestion_text[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error mostrando sugerencia: {e}")
    
    def show(self):
        """Muestra el overlay"""
        if self.root and not self.is_visible:
            self.root.deiconify()
            self.is_visible = True
            logger.debug("👁️ Overlay visible")
    
    def hide(self):
        """Oculta el overlay"""
        if self.root and self.is_visible:
            self.root.withdraw()
            self.is_visible = False
            self._cancel_auto_hide_timer()
            logger.debug("🙈 Overlay oculto")
    
    def _start_auto_hide_timer(self):
        """Inicia timer para auto-ocultar"""
        self._cancel_auto_hide_timer()
        
        self.auto_hide_timer = threading.Timer(
            self.config.auto_hide_delay,
            self._auto_hide
        )
        self.auto_hide_timer.start()
        
        # Actualizar contador visual
        self._update_timer_display()
    
    def _cancel_auto_hide_timer(self):
        """Cancela timer de auto-ocultar"""
        if self.auto_hide_timer:
            self.auto_hide_timer.cancel()
            self.auto_hide_timer = None
        
        if self.timer_label:
            self.timer_label.config(text="")
    
    def _auto_hide(self):
        """Auto-oculta el overlay"""
        self.hide()
        logger.debug("⏰ Auto-hide triggered")
    
    def _update_timer_display(self):
        """Actualiza el display del timer"""
        if not self.auto_hide_timer or not self.is_visible:
            return
        
        if self.auto_hide_timer.is_alive():
            # Calcular tiempo restante aproximado
            remaining = int(self.config.auto_hide_delay)  # Simplificado
            self.timer_label.config(text=f"🕐 {remaining}s")
            
            # Programar siguiente actualización
            self.root.after(1000, self._update_timer_display)
    
    def _on_close_clicked(self):
        """Maneja click en botón cerrar"""
        self.hide()
        
        if self.on_close_callback:
            try:
                self.on_close_callback()
            except Exception as e:
                logger.error(f"❌ Error en close callback: {e}")
    
    def _on_click(self, event):
        """Maneja clicks en el overlay"""
        # Permitir arrastrar la ventana
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        
        self.root.bind("<B1-Motion>", self._on_drag)
        self.root.bind("<ButtonRelease-1>", self._on_drag_end)
    
    def _on_drag(self, event):
        """Maneja arrastrar ventana"""
        if hasattr(self, '_drag_start_x'):
            x = self.root.winfo_x() + (event.x_root - self._drag_start_x)
            y = self.root.winfo_y() + (event.y_root - self._drag_start_y)
            self.root.geometry(f"+{x}+{y}")
            
            self._drag_start_x = event.x_root
            self._drag_start_y = event.y_root
    
    def _on_drag_end(self, event):
        """Termina arrastrar"""
        self.root.unbind("<B1-Motion>")
        self.root.unbind("<ButtonRelease-1>")
        
        if hasattr(self, '_drag_start_x'):
            delattr(self, '_drag_start_x')
            delattr(self, '_drag_start_y')
    
    def _on_close_hover(self, event):
        """Hover effect en botón cerrar"""
        self.close_button.config(fg="#FF6B6B")  # Rojo al hover
    
    def _on_close_leave(self, event):
        """Leave effect en botón cerrar"""
        self.close_button.config(fg=self.config.text_color)
    
    def set_close_callback(self, callback: Callable):
        """Establece callback para cuando se cierra"""
        self.on_close_callback = callback
    
    def run_mainloop(self):
        """Ejecuta el mainloop de tkinter"""
        if self.root:
            self.root.mainloop()
    
    def destroy(self):
        """Destruye el overlay"""
        self._cancel_auto_hide_timer()
        
        if self.root:
            self.root.destroy()
            self.root = None
        
        logger.info("🗑️ Overlay destruido")


# Factory y manager para uso fácil
class OverlayManager:
    """Manager para manejar overlay en thread separado"""
    
    def __init__(self):
        self.overlay: Optional[SuggestionOverlay] = None
        self.overlay_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
    def start(self):
        """Inicia el overlay en thread separado"""
        self.overlay_thread = threading.Thread(
            target=self._run_overlay,
            daemon=True
        )
        self.overlay_thread.start()
        
        # Esperar a que se inicialice
        time.sleep(0.5)
        
        logger.success("🚀 Overlay manager iniciado")
    
    def _run_overlay(self):
        """Ejecuta overlay en thread separado"""
        try:
            self.overlay = SuggestionOverlay()
            self.overlay.initialize()
            self.overlay.run_mainloop()
        except Exception as e:
            logger.error(f"❌ Error en overlay thread: {e}")
    
    def show_suggestion(self, text: str, context: str = "", confidence: float = 0.8):
        """Thread-safe method para mostrar sugerencia"""
        if self.overlay and self.overlay.root:
            # Ejecutar en el thread del UI
            self.overlay.root.after(0, 
                lambda: self.overlay.show_suggestion(text, context, confidence)
            )
    
    def hide(self):
        """Thread-safe method para ocultar"""
        if self.overlay and self.overlay.root:
            self.overlay.root.after(0, self.overlay.hide)
    
    def stop(self):
        """Detiene el overlay"""
        if self.overlay:
            if self.overlay.root:
                self.overlay.root.after(0, self.overlay.destroy)


# Factory para configuración optimizada
def create_lean_overlay() -> OverlayManager:
    """Crea overlay con configuración lean"""
    return OverlayManager()


if __name__ == "__main__":
    """Test del overlay UI"""
    
    def test_overlay():
        logger.info("🧪 Testing overlay UI...")
        
        manager = create_lean_overlay()
        manager.start()
        
        # Test después de 2 segundos
        time.sleep(2)
        
        manager.show_suggestion(
            "Entiendo tu preocupación sobre el timeline. ¿Podríamos revisar las prioridades juntos para encontrar una solución que funcione para todos?",
            "Basado en 2 ejemplos similares",
            0.87
        )
        
        # Mantener vivo por 15 segundos
        time.sleep(15)
        
        manager.stop()
        logger.success("✅ Test overlay completado")
    
    test_overlay()
