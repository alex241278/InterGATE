"""Publication typography only; no model, data, selection or training imports."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DEFAULT_STYLE = {
    'font_family': 'DejaVu Sans', 'base_font_size': 14.0,
    'figure_title_size': 20.0, 'panel_title_size': 16.0,
    'axis_label_size': 14.0, 'tick_label_size': 13.0,
    'legend_size': 12.0, 'annotation_size': 12.0,
    'gene_label_size': 12.5, 'dpi': 600,
    'svg_fonttype': 'none', 'pdf_fonttype': 42,
}


def load_style(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    path = root / 'configs' / 'publication_style.json'
    style = DEFAULT_STYLE.copy()
    if path.is_file():
        updates = json.loads(path.read_text(encoding='utf-8'))
        unknown = set(updates) - set(style)
        if unknown:
            raise ValueError(f'Unknown publication-style fields: {sorted(unknown)}')
        style.update(updates)
    for key, value in style.items():
        if key.endswith('_size') or key == 'dpi':
            if not isinstance(value, (int, float)) or not 0 < value < 10000:
                raise ValueError(f'Invalid style value: {key}={value!r}')
    return style


def apply_style(style: dict[str, Any] | None = None) -> dict[str, Any]:
    import matplotlib as mpl
    style = load_style() if style is None else style
    mpl.rcParams.update({
        'font.family': style['font_family'], 'font.size': style['base_font_size'],
        'axes.titlesize': style['panel_title_size'], 'axes.titleweight': 'bold',
        'axes.labelsize': style['axis_label_size'],
        'xtick.labelsize': style['tick_label_size'],
        'ytick.labelsize': style['tick_label_size'],
        'legend.fontsize': style['legend_size'],
        'svg.fonttype': style['svg_fonttype'], 'pdf.fonttype': style['pdf_fonttype'],
    })
    return style


def prepare_figure(fig, style: dict[str, Any] | None = None) -> None:
    """Raise small explicit annotations too. Never alter data, colours or limits.

    Idempotent: calling this before PDF, SVG and PNG export does not multiply
    the font sizes. Layout must be inspected at the final manuscript width.
    """
    from matplotlib.text import Text
    style = load_style() if style is None else style
    for text in fig.findobj(match=Text):
        if text.get_visible() and text.get_text().strip():
            text.set_fontsize(max(text.get_fontsize(), style['annotation_size']))
    for ax in fig.axes:
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(max(title.get_fontsize(), style['panel_title_size']))
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(max(label.get_fontsize(), style['axis_label_size']))
        ax.tick_params(axis='both', labelsize=style['tick_label_size'])
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(max(text.get_fontsize(), style['legend_size']))
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(max(fig._suptitle.get_fontsize(), style['figure_title_size']))
    # Existing constrained/tight layout engines will account for the new fonts.
    fig.canvas.draw()
