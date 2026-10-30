# Assets Directory

This directory contains visual assets for the repository.

## 📸 Required Images

For the main README, please add:

### 1. `pind_overview.png` (800px wide)
- Overview diagram showing the PIND architecture
- Should include: robot, physics engine, neural networks
- Recommended tools: draw.io, PowerPoint, Inkscape, Figma

### 2. `data_efficiency_plot.png` (600px wide)
- Line plot showing data efficiency comparison
- X-axis: Data percentage (5%, 10%, 50%, 100%)
- Y-axis: Position RMSE (mm)
- Lines for each method with error bars

### Example Python code to generate plots:

```python
import matplotlib.pyplot as plt
import numpy as np

# Data efficiency plot
data_fractions = [5, 10, 50, 100]
pind_errors = [12.4, 8.7, 4.2, 2.8]
pind_std = [2.1, 1.5, 0.9, 0.6]

mlp_errors = [45.3, 38.1, 22.4, 15.6]
node_errors = [42.1, 35.8, 21.2, 14.8]
residual_errors = [28.4, 22.4, 12.6, 8.9]

plt.figure(figsize=(10, 6))
plt.errorbar(data_fractions, pind_errors, yerr=pind_std, 
             marker='o', linewidth=2, capsize=5, label='PIND (Ours)')
plt.plot(data_fractions, mlp_errors, marker='s', linewidth=2, label='Black-box MLP')
plt.plot(data_fractions, node_errors, marker='^', linewidth=2, label='Neural ODE')
plt.plot(data_fractions, residual_errors, marker='d', linewidth=2, label='Residual MLP')

plt.xlabel('Training Data (%)', fontsize=14)
plt.ylabel('50-Step Position RMSE (mm)', fontsize=14)
plt.title('Data Efficiency Comparison', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/data_efficiency_plot.png', dpi=300)
```

## 🎨 Creating Professional Figures

### Tools
- **draw.io**: Free, online, great for architecture diagrams
- **Figma**: Professional design tool
- **Inkscape**: Free vector graphics editor
- **PowerPoint/Keynote**: Quick mockups
- **Python/Matplotlib**: Data plots

### Style Guide
- Use consistent colors across all figures
- Include legends and axis labels
- Export at 300 DPI for publication quality
- Keep background clean (white or transparent)
- Use clear, readable fonts (Arial, Helvetica, or Source Sans Pro)

## 📐 Recommended Dimensions

- Overview diagram: 800-1200px wide
- Result plots: 600-800px wide
- Icons/logos: 64-128px
- Architecture diagrams: 800-1000px wide

## 🖼️ Placeholder Images

Until you create the actual images, you can use placeholder services:
- https://placehold.co/800x400/png?text=PIND+Overview
- https://placehold.co/600x400/png?text=Results+Plot

Or create simple diagrams with ASCII art in code blocks for now.

## 📦 Organization

```
assets/
├── pind_overview.png           # Main architecture diagram
├── data_efficiency_plot.png    # Results comparison
├── architecture_detail.png     # Detailed architecture
├── robot_photo.png             # Robot hardware (if available)
└── demo.gif                    # Demo animation (optional)
```

---

**Note:** All images should be optimized for web (compressed, appropriate size).
GitHub automatically serves images, so keep file sizes reasonable (< 2MB each).
