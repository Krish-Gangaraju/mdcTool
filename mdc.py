import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re


@st.cache_data
def clean_strain_sweep(buffer):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('latin-1', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]

    new_cols = ['Temp1', 'dyn Str1', 'Freq1', 'Temp2', 'Freq2', 'dyn D', 'stat D', 'dyn Str', 'sStrain',
        'dyn F (N)', 'dyn C (N/m\u00B2)', 'G (Pa)', "G' (Pa)", "G'' (Pa)", 'Tan Delta']

    cleaned = []
    blank_count = 0
    for ln in lines[22:]:
        if not ln.strip():
            blank_count += 1
            if blank_count > 1:
                break
        else:
            blank_count = 0
            cleaned.append(ln.strip())

    rows = [ln.lstrip(';').split(';') for ln in cleaned]
    df = pd.DataFrame(rows, columns=new_cols)

    x_col  = 'dyn Str'
    metrics = ["G' (Pa)", "G'' (Pa)", "Tan Delta"]

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        df[f"{m}_smooth"] = df[m].rolling(window=3, center=True).mean()

    # compute G* = sqrt(G'^2 + G''^2) and then smooth it
    gstar = np.sqrt(df["G' (Pa)"]**2 + df["G'' (Pa)"]**2)
    df.insert(loc=14, column="G* (Pa)", value=gstar)
    df["G* (Pa)_smooth"] = df["G* (Pa)"].rolling(window=3, center=True).mean() 

    return df



@st.cache_data
def clean_temp_sweep(buffer):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('latin-1', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]

    new_cols = ['dyn F (N)', 'Freq (Hz)', 'Time (s)', "G' (Pa)", "G'' (Pa)", 'Tan Delta', 'Temperature (°C)', 'dyn Str', 'Time1']

    cleaned = []
    blank_count = 0
    for ln in lines[16:]:
        if not ln.strip():
            blank_count += 1
            if blank_count > 1:
                break
        else:
            blank_count = 0
            cleaned.append(ln.strip())

    rows = [ln.lstrip(';').split(';') for ln in cleaned]
    df = pd.DataFrame(rows, columns=new_cols)

    x_col  = 'Temperature (°C)'
    metrics = ["G' (Pa)", "G'' (Pa)", "Tan Delta"]

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        df[f"{m}_smooth"] = df[m].rolling(window=20, center=True).mean()

    # compute G* = sqrt(G'^2 + G''^2) and then smooth it
    gstar = np.sqrt(df["G' (Pa)"]**2 + df["G'' (Pa)"]**2)
    df.insert(loc=5, column="G* (Pa)", value=gstar)
    df["G* (Pa)_smooth"] = df["G* (Pa)"].rolling(window=20, center=True).mean()

    return df




# ——— Streamlit UI ———
st.set_page_config(layout="wide")
st.title("MDC Post-Processing Tool")

mode = st.selectbox("Choose sweep type:", ["Strain Sweep", "Temperature Sweep"])

uploaded = st.file_uploader("Upload one or more files",  type=['asc'], accept_multiple_files=True,
    key="uploader_strain" if mode=="Strain Sweep" else "uploader_temp")

if not uploaded:
    st.info("📂 Please upload at least one file to continue.")
    st.stop()

processed = {}
for f in uploaded:
    try:
        processed[f.name] = clean_strain_sweep(f) if mode=="Strain Sweep" else clean_temp_sweep(f)
    except Exception as e:
        st.error(f"⚠️ Failed loading **{f.name}**: {e}")
if not processed:
    st.stop()

st.subheader("Select mixes to plot")
mixes = []
for name in sorted(processed):
    label = name.rsplit('.',1)[0]
    if st.checkbox(label, value=True, key=f"cb_{name}"):
        mixes.append(name)
if not mixes:
    st.info("Select at least one mix to see the panels.")
    st.stop()

tab_graph, tab_key, tab_data = st.tabs(["Graph Interface","Key Values","Data Interface"])



# — Graph Interface —
with tab_graph:
    st.subheader(f"{mode} Graphs")

    if mode == "Strain Sweep":
        phase = st.radio("Phase", ["Both","Go","Return"], horizontal=True)
        x_axis  = "dyn Str"
    elif mode == "Temperature Sweep":
        phase = "Both"
        x_axis  = "Temperature (°C)"
        
    metrics = ["G' (Pa)", "G'' (Pa)", "G* (Pa)", "Tan Delta"]

    cols = st.columns(2, gap="large")
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots()
        for name in mixes:
            df = processed[name].copy()

            if mode=="Strain Sweep":
                peak = df[x_axis].idxmax()
                if phase=="Go":
                    df = df.loc[:peak]
                elif phase=="Return":
                    df = df.loc[peak:]

                df = df[df[x_axis]>0]

                y = df[f"{metric}_smooth"] if metric!="G* (Pa)" else df["G* (Pa)_smooth"]
                if "(Pa)" in metric:
                    y *= 1e-6
                    y_label = metric.replace("(Pa)","(MPa)")
                else:
                    y_label = metric

                ax.plot(df[x_axis], y, label=name.rsplit('.',1)[0], linewidth=1.5)

            else:
                deg    = df[x_axis].round().astype(int)
                y_raw  = df[f"{metric}_smooth"] if metric!="G* (Pa)" else df["G* (Pa)_smooth"]
                df2    = pd.DataFrame({x_axis: deg, 'y': y_raw})
                avg    = df2.groupby(x_axis)['y'].mean().reset_index()
                avg[f"{metric}_agg_smooth"] = avg['y'].rolling(window=20, center=True).mean()
                x_vals = avg[x_axis]
                y_vals = avg[f"{metric}_agg_smooth"]
                if "(Pa)" in metric:
                    y_vals *= 1e-6
                    y_label = metric.replace("(Pa)","(MPa)")
                else:
                    y_label = metric

                ax.plot(x_vals, y_vals, label=name.rsplit('.',1)[0], linewidth=1.5)

        if mode=="Strain Sweep":
            xvals = df[x_axis]
            min_e = int(np.floor(np.log10(xvals.min())))
            max_e = int(np.ceil(np.log10(xvals.max())))
            ticks = [10**e for e in range(min_e, max_e+1)]
            ax.set_xscale('log')
            if metric == "Tan Delta":
                ax.set_xticks(ticks + [10])
                ax.set_xlim(ticks[0], 10)
            else:
                ax.set_xticks(ticks)
                ax.set_xlim(ticks[0], ticks[-1])
        else:
            ax.set_xlim(df[x_axis].min(), df[x_axis].max())

        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_label)
        if mode == "Strain Sweep":
            ax.set_title(f"{y_label} vs {x_axis} ({phase})")
        else:
            ax.set_title(f"{y_label} vs {x_axis}")
        ax.legend(fontsize="small", loc="best")
        cols[i%2].pyplot(fig, use_container_width=True)




# — Key Values Interface —
with tab_key:
    st.subheader("Key Values")

    # choose the appropriate set of keys
    if mode == "Strain Sweep":
        key_names = [
            "Tempe",
            "Machine #",
            "% Accomodation = Accomodation / G* Go",
            "Accomodation = G* DEF MIN Go - Return",
            "Calcul NL = G*DEF MIN - G*DEF MAX Go",
            "Calcul NL = G*DEF MIN - G*DEF MAX Return",
            "DEFORMATION G\" MAX (%) Go",
            "DEFORMATION G\" MAX (%) Return",
            "DEFORMATION Tg MAX Go (%)",
            "DEFORMATION Tg MAX Return (%)",
            "G' 10% Return",
            "G' 35% Return (MPa)",
            "G' DEF MIN Go (MPa)",
            "G' DEF MIN Return (MPa)",
            "G' DEF MAX Go (MPa)",
            "G' DEF MAX Return (MPa)",
            "G'' 10% Go (MPa)",
            "G'' 10% Return (MPa)",
            "G'' 20% Return (MPa)",
            "G'' 35% Return (MPa)",
            "G'' MAX Return (MPa)",
            "G'' DEF MIN Return (MPa)",
            "G* 0.2% Go (MPa)",
            "G* 0.2% Return (MPa)",
            "G* 0.6% Return (MPa)",
            "G* 1% Go (MPa)",
            "G* 1% Return (MPa)",
            "G* 2% Go (MPa)",
            "G* 2% Return (MPa)",
            "G* 10% Go (MPa)",
            "G* 10% Return (MPa)",
            "G* 15% Go (MPa)",
            "G* 15% Return (MPa)",
            "G* 20% Go (MPa)",
            "G* 20% Return (MPa)",
            "G* 25% Go (MPa)",
            "G* 25% Return (MPa)",
            "G* 50% Go (MPa)",
            "G* 50% Return (MPa)",
            "G* 97% Go (MPa)",
            "G* 97% Return (MPa)",
            "G* DEF MIN Go (MPa)",
            "G* DEF MIN Return (MPa)",
            "G* DEF MAX Go (MPa)",
            "G* DEF MAX Return (MPa)",
            "Intégrale G\" Return",
            "Intégrale tand Return",
            "Tan D - 0.1% Return",
            "Tan D - 1% Go",
            "Tan D - 1% Return",
            "Tan D - 10% Go",
            "Tan D - 10% Return",
            "Tan D - 20% Go",
            "Tan D - 20% Return",
            "Tan D MAX Return",
            "Tan D DEF MAX Return",
            "Tan D DEF MIN Return",
            "Tan D MAX Go",
            "G* def MAX / G* def MIN"
        ]
    elif mode == "Temperature Sweep":
        key_names = [
            "Stress",
            "Machine #",
            "Def % -20°C",
            "Def % -10°C",
            "Def % 10°C",
            "Def % 30°C",
            "Def % 60°C",
            "Def % 100°C",
            "G'' 10°C (MPa)",
            "G'' 90°C (MPa)",
            "G'' MAX (MPa)",
            "G* -30°C (MPa)",
            "G* -20°C (MPa)",
            "G* -10°C (MPa)",
            "G* 0°C (MPa)",
            "G* 10°C (MPa)",
            "G* 20°C (MPa)",
            "G* 30°C (MPa)",
            "G* 40°C (MPa)",
            "G* 50°C (MPa)",
            "G* 60°C (MPa)",
            "G* 90°C (MPa)",
            "G* 100°C (MPa)",
            "G''/G*² -20°C",
            "G''/G*² MAX",
            "Integrale G''",
            "Integrale Tan D",
            "Integrale [0°C.25°C]",
            "Integrale Tan [-70°C.0°C]",
            "Integrale Tan [-70°C.-30°C]",
            "Integrale Tan [-40°C.50°C]",
            "Integrale Tan [-37°C.10°C]",
            "Integrale Tan [-10°C.10°C]",
            "Integrale Tan [-10°C.25°C]",
            "J'' -30°C (MPa⁻¹)",
            "J'' -20°C (MPa⁻¹)",
            "J'' -10°C (MPa⁻¹)",
            "J'' 60°C (MPa⁻¹)",
            "Slope (G*98°C-G*75°C)/ΔT",
            "T (°C) G'' MAX",
            "T (°C) G''/G*² MAX",
            "T (°C) Tan D MAX",
            "Tan -30°C",
            "Tan -20°C",
            "Tan -10°C",
            "Tan 0°C",
            "Tan 10°C",
            "Tan 20°C",
            "Tan 30°C",
            "Tan 40°C",
            "Tan 50°C",
            "Tan 60°C",
            "Tan 90°C",
            "Tan 100°C",
            "Tan MAX",
            "Tan Elastomer (°C)",
            "Temp (°C) G*=1.5MPa",
            "Temp (°C) G*=3MPa",
            "Temp (°C) G*=5MPa",
            "Temp (°C) G*=10MPa",
            "Temp (°C) G*=100MPa"
        ]

    # file names as columns, without extension
    cols = [name.rsplit('.',1)[0] for name in sorted(processed.keys())]

    # assemble empty DataFrame
    summary_df = pd.DataFrame(index=key_names, columns=cols)
    st.dataframe(summary_df, use_container_width=True, height=600)




# — Data Interface —
with tab_data:
    st.subheader("Raw Data Tables")
    for name in mixes:
        st.markdown(f"**{name.rsplit('.',1)[0]}**")
        df = processed[name]

        # drop the “_smooth” columns as before
        drop_cols = [f"{m}_smooth" for m in ["G' (Pa)","G'' (Pa)","Tan Delta","G* (Pa)"]]
        df_display = df.drop(columns=drop_cols).copy()

        # find all G-columns that end in “(Pa)”
        g_cols = [c for c in df_display.columns if re.match(r"^G.*\(Pa\)$", c)]

        # convert them to numeric and scale from Pa → MPa
        for c in g_cols:
            df_display[c] = pd.to_numeric(df_display[c], errors='coerce') / 1e6

        # rename the headers from “(Pa)” → “(MPa)”
        rename_map = {c: c.replace("(Pa)", "(MPa)") for c in g_cols}
        df_display = df_display.rename(columns=rename_map)
        # df_display["diff"] = np.sqrt(df_display["G* (MPa)"]**2 - df_display["G' (MPa)"]**2)

        st.dataframe(df_display, use_container_width=True)

