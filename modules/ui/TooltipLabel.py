import tkinter as tk
from tkinter import ttk

class TooltipLabel(ttk.Label):
    """
    A label with a tooltip that appears when hovering over it.
    """
    def __init__(self, master=None, text="", tooltip="", **kwargs):
        super().__init__(master, text=text, **kwargs)
        self.tooltip = tooltip
        self.tooltip_window = None
        
        self.bind("<Enter>", self._show_tooltip)
        self.bind("<Leave>", self._hide_tooltip)
    
    def _show_tooltip(self, event=None):
        """Show the tooltip when the mouse enters the widget"""
        if self.tooltip:
            x, y, _, _ = self.bbox("insert")
            x += self.winfo_rootx() + 25
            y += self.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip_window = tk.Toplevel(self)
            self.tooltip_window.wm_overrideredirect(True)
            self.tooltip_window.wm_geometry(f"+{x}+{y}")
            
            # Create tooltip label
            label = ttk.Label(self.tooltip_window, text=self.tooltip, 
                             background="#ffffe0", relief="solid", borderwidth=1,
                             wraplength=250, justify="left", padding=(5, 2))
            label.pack()
    
    def _hide_tooltip(self, event=None):
        """Hide the tooltip when the mouse leaves the widget"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None