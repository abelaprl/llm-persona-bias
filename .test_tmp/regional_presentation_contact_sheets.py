"""Create inspection-only contact sheets from the frozen scientific figures."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('bundle', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
index = pd.read_csv(args.bundle / 'figure_index.csv')
for offset in range(0, len(index), 12):
    page = index.iloc[offset:offset + 12]
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), dpi=120)
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, page.itertuples(index=False)):
        with Image.open(args.bundle / row.png) as original:
            thumbnail = original.copy()
        thumbnail.thumbnail((550, 580))
        ax.imshow(thumbnail)
        ax.set_title(row.figure_id, fontsize=8)
    fig.tight_layout()
    path = args.output / f'contact_{offset // 12 + 1:02d}.png'
    fig.savefig(path)
    plt.close(fig)
    print(path, flush=True)
