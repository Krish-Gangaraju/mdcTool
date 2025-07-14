# mdcTool


### 🚀 Getting Started

1. **Launch the app** in your browser.  
2. **Choose your sweep type** from the top dropdown:  
   - **Strain Sweep**  
   - **Temperature Sweep**  
   - **Help & User Manual** (this guide)  
3. **Upload your data**  
   - Drag & drop one or more `.asc` files into the uploader.  
   - Each file becomes one “mix”—you can process up to 20 at once.  
   - Wait a second for the tool to clean and cache your data.  
4. **Work in the three tabs** that appear:  
   - **Graph Interface**  
   - **Key Values**  
   - **Data Interface**

---

## 🖥️ 1. Graph Interface

#### Strain Sweep mode

- **Phase selector** (“Both” / “Go” / “Return”)  
- **Grid toggle** (On/Off)  
- **Mix checkboxes**—turn individual curves on or off  
- **Four panels** (2×2 layout) plotting:  
  1. G′ vs Strain  
  2. G″ vs Strain  
  3. G*  vs Strain  
  4. Tan Delta vs Strain  
- **Custom titles** for each panel  
- **Log-scale X-axis** with human-readable tick labels  
- **Auto-padded Y-limits** so curves never hug the edges  
- **Download** any finished plot as a high-res PNG  

#### Temperature Sweep mode

- **Grid toggle** (On/Off)  
- **Mix checkboxes**—select which files to include  
- For each **mechanical metric** (G′, G″, G*, Tan Delta):  
  - Side-by-side plots—one per force level in your file  
  - **Custom titles** for each subplot  
  - Linear X (Temperature) and Y (metric) axes  
- **Download** each temperature-sweep plot as PNG  

---

## 🔑 2. Key Values

#### Strain Sweep

Automatically computes for each mix:

- **Peak & threshold points**  
  - G′/G″ at standard strain levels (0.1%, 0.5%, 1%, 2%, 10%, etc.)  
  - Tan Delta max and values at 1% & 10% strain  

Displays a table you can **sort**, **scroll**, and **export** to Excel.

#### Temperature Sweep

For each force level:

- **Metric values** at key temperatures (–30 °C, 0 °C, 30 °C, 90 °C, etc.)  
- **Maxima** (e.g. the temperature where Tan Delta peaks)  
- **Slopes** over important spans (e.g. G* between 75 °C–98 °C)  
- **Temperatures** at which G* crosses standard thresholds (1.5 MPa, 5 MPa, 10 MPa…)  

Each “dyn F” block has its own **Download** button for a ready-made Excel sheet.

---

## 📂 3. Data Interface

#### Strain Sweep

- View the **cleaned** DataFrame for each mix  
- “_smooth” columns hidden; moduli shown in MPa  
- **Interactive** filter, sort, and scroll  

#### Temperature Sweep

- **Download** raw “mini-tests” as an Excel workbook (one sheet per force level)  
- **Preview** the first sheet right in your browser  

---

## ⚙️ Common Controls & Tips

- **Select All** checkbox to toggle every mix on/off.  
- **Live caching**: re-upload the same file instantly.  
- **Consistent color-coding** across plots for easy tracking.  
- **No coding required**—everything is point-and-click.  

---
