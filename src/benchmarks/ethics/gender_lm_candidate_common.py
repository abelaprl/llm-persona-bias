"""Shared, additive exports for the two frozen-source candidate presentations."""
from pathlib import Path

import matplotlib.pyplot as plt

from .gender_lm_presentation_v2_plots import PRINT_STYLE, audit_geometry


def save_panel(root, stem, title, question, caption, fig, table, section='main'):
    """Export a print-size figure and its exact display table, never replacing files."""
    root = Path(root)
    if Path(stem).name != stem or not stem or stem in {'.', '..'}:
        raise ValueError('The figure stem must be a simple name')
    paths = {'png': root / 'figures' / (stem + '.png'),
             'svg': root / 'figures' / (stem + '.svg'),
             'csv': root / 'tables' / (stem + '.csv')}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(path)
    geometry = audit_geometry(fig)
    if abs(geometry['width_cm'] - 16.) > .01:
        raise ValueError('Candidate figures must be 16 cm wide')
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths['png'], dpi=240)
    fig.savefig(paths['svg'])
    table.to_csv(paths['csv'], index=False)
    plt.close(fig)
    return dict(id=stem, title=title, question=question, caption=caption,
                section=section, geometry=geometry,
                **{key: path.relative_to(root).as_posix() for key, path in paths.items()})
