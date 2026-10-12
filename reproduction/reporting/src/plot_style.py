"""Shared display settings preserved from the existing quantitative figures."""
from matplotlib.transforms import ScaledTranslation
MODELS = ("state_front", "static_front", "rtu", "fno")
NAMES = {"state_front": "DynaRO", "static_front": "Static front", "rtu": "RTU", "fno": "FNO"}
COLORS = {"state_front": "#428F7D", "static_front": "#C97C9D",
          "rtu": "#8A8F9E", "fno": "#AA9B83"}
STYLES = {"state_front": "-", "static_front": "--", "rtu": "-.", "fno": ":"}

def panel_label(fig, ax, letter, title):
    transform = ax.transAxes + ScaledTranslation(-14/72, 9/72, fig.dpi_scale_trans)
    ax.text(0, 1, letter, transform=transform, va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title, loc="left", pad=10, fontsize=8.4, fontweight="semibold")
