import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import re
import io



@st.cache_data
def clean_strain_sweep(buffer):
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('latin-1', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]

    new_cols = ['Temp1', 'dyn Str1', 'Freq1', 'Temp2', 'Freq2', 'dyn D', 'stat D', 'dyn Str', 'sStrain',
                'dyn F (N)', 'dyn C (N/m\u00B2)', 'G', "G'", "G''", 'Tan Delta']

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

    x_col = 'dyn Str'
    metrics = ["G'", "G''", "Tan Delta"]
    window_size = 5

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        sm = f"{m}_smooth"
        df[sm] = df[m].rolling(window=window_size, center=True).mean()
        # fill leading NaNs
        first = df[sm].first_valid_index()
        if first is not None and first > 0:
            df.loc[:first-1, sm] = df.loc[:first-1, m]
        # fill trailing NaNs
        last = df[sm].last_valid_index()
        if last is not None and last < len(df)-1:
            df.loc[last+1:, sm] = df.loc[last+1:, m]

    # compute G* = sqrt(G'^2 + G''^2) and then smooth it
    df["G*"] = np.sqrt(df["G'"]**2 + df["G''"]**2)
    df["G*_smooth"] = df["G*"].rolling(window=window_size, center=True).mean()
    first_g = df["G*_smooth"].first_valid_index()
    if first_g is not None and first_g > 0:
        df.loc[:first_g-1, "G*_smooth"] = df.loc[:first_g-1, "G*"]
    last_g = df["G*_smooth"].last_valid_index()
    if last_g is not None and last_g < len(df)-1:
        df.loc[last_g+1:, "G*_smooth"] = df.loc[last_g+1:, "G*"]

    return df



@st.cache_data
def clean_temp_sweep(buffer):
    # 1) read & decode whole file
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('latin-1', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]

    # 2) pull out data rows
    new_cols = [
        'dyn F (N)', 'Freq (Hz)', 'Time (s)', "G'", "G''", 'Tan Delta',
        'Temperature (°C)', 'dyn Str', 'Time1'
    ]
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

    # 3) split into mini-tests by dyn F
    mini_tests = {}
    for val in df['dyn F (N)'].unique():
        mini_tests[float(val)] = df[df['dyn F (N)'] == val].copy()

    # 4) for each mini-test: convert, compute G*, smooth & fill edges
    window = 5
    for f, sub in mini_tests.items():
        sub['Temperature (°C)'] = pd.to_numeric(sub['Temperature (°C)'], errors='coerce')
        for m in ["G'", "G''", "Tan Delta"]:
            sub[m] = pd.to_numeric(sub[m], errors='coerce')
        sub["G*"] = np.sqrt(sub["G'"]**2 + sub["G''"]**2)

        for col in ["G'", "G''", "Tan Delta", "G*"]:
            sm_col = f"{col}_smooth"
            sub[sm_col] = sub[col].rolling(window=window, center=True).mean()

            # fill leading NaNs with raw
            first = sub[sm_col].first_valid_index()
            if first is not None and first > 0:
                sub.loc[:first-1, sm_col] = sub.loc[:first-1, col]

            # fill trailing NaNs with raw
            last = sub[sm_col].last_valid_index()
            if last is not None and last < len(sub)-1:
                sub.loc[last+1:, sm_col] = sub.loc[last+1:, col]

        mini_tests[f] = sub

    return mini_tests



@st.cache_data
def aggregate_by_degree(df: pd.DataFrame, degree_col: str, smooth_cols: list[str]) -> pd.DataFrame:
    df_ = df.copy()
    df_['deg'] = df_[degree_col].round().astype(int)
    # group by that integer degree, average all the smoothed metrics
    agg = (df_.groupby('deg')[smooth_cols].mean().reset_index().rename(columns={'deg': degree_col}))
    return agg


# ——— Streamlit  ———
st.set_page_config(page_title="MDC Post-Processing Tool", layout="wide")
st.title("MDC Post-Processing Tool")

mode = st.selectbox("Choose sweep type:", ["Strain Sweep", "Temperature Sweep"])

uploaded = st.file_uploader("Upload files",  type=['asc'], accept_multiple_files=True, key="uploader_strain" if mode=="Strain Sweep" else "uploader_temp")
 
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

if mode=="Temperature Sweep":
    # for each file (name) and each mini‐test (dynF → df):
    raw_force_aggs = {
        name: {
            dynF: (
                df.assign(deg=df['Temperature (°C)'].round().astype(int), dynF_num=pd.to_numeric(df['dyn F (N)'], errors='coerce'))
                .groupby('deg')['dynF_num'].mean()
            )
            for dynF, df in tests.items()
        }
        for name, tests in processed.items()
    }

    raw_val_aggs = {
        name: {
            dynF: (
                df.assign(deg=df['Temperature (°C)'].round().astype(int)).groupby('deg')[["G''","G*","Tan Delta"]].mean()
            )
            for dynF, df in tests.items()
        }
        for name, tests in processed.items()
    }


tab_graph, tab_key, tab_data = st.tabs(["Graph Interface","Key Values","Data Interface"])



# — Graph Interface —
with tab_graph:

    if mode == "Strain Sweep":
        st.subheader("Strain Sweep Graphs")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            phase = st.radio("Phase", ["Both", "Go", "Return"], horizontal=True)
        with col2:
            grid_on = st.radio("Grid lines", ["On", "Off"], horizontal=True) == "On"

        st.markdown("**Select mixes to plot**")
        mixes = []
        for name in sorted(processed):
            label = name.rsplit('.', 1)[0]
            if st.checkbox(label, value=True, key=f"cb_graph_{name}"):
                mixes.append(name)
        if not mixes:
            st.info("Select at least one mix to see the panels.")
            st.stop()

        metrics = ["G'", "G''", "G*", "Tan Delta"]
        # render in 2×2 grid, each with its own title input
        for row in (metrics[:2], metrics[2:]):
            plot_cols = st.columns(2, gap="large")
            for metric, pc in zip(row, plot_cols):
                with pc:
                    title = st.text_input(
                        f"**Title for {metric}**",
                        value=f"{metric} vs Strain ({phase})",
                        key=f"title_{metric}"
                    )

                    # start the figure & axes
                    fig, ax = plt.subplots()

                    # we'll track the global y‐min/max across all mixes
                    global_y_min = float('inf')
                    global_y_max = float('-inf')

                    # plot each mix
                    for name in mixes:
                        df = processed[name].copy()
                        peak = df["dyn Str"].idxmax()
                        if phase == "Go":
                            df = df.loc[:peak]
                        elif phase == "Return":
                            df = df.loc[peak:]
                        df = df[df["dyn Str"] > 0]

                        y = df[f"{metric}_smooth"] if metric != "G*" else df["G*_smooth"]
                        if metric in ("G'", "G''", "G*"):
                            y = y * 1e-6

                        # update our global y‐bounds
                        if not y.empty:
                            global_y_min = min(global_y_min, y.min())
                            global_y_max = max(global_y_max, y.max())

                        ax.plot(
                            df["dyn Str"],
                            y,
                            label=name.rsplit('.', 1)[0],
                            linewidth=1.5
                        )

                    # force exponent range from 10^-3 upward (same as before)
                    min_e = -3
                    max_e = int(np.ceil(np.log10(
                        max(processed[name]["dyn Str"].max() for name in mixes)
                    )))
                    ticks = [10**e for e in range(min_e, max_e + 1)]
                    ax.set_xscale('log')
                    ax.set_xticks(ticks)
                    ax.set_xlim(ticks[0], ticks[-1])
                    ax.set_xticklabels([f"{t:g}" for t in ticks])

                    # apply a 50% margin based on the global max y
                    padding = 0.4 * global_y_max
                    y_lower = global_y_min - padding
                    y_upper = global_y_max + padding

                    ax.set_ylim(y_lower, y_upper)
                    ax.grid(grid_on)
                    ax.set_xlabel("Strain [%]")

                    unit = " [MPa]" if metric in ("G'", "G''", "G*") else ""
                    ax.set_ylabel(metric + unit)
                    ax.set_title(title)
                    ax.legend(fontsize="small", loc="best")

                    pc.pyplot(fig, use_container_width=True)



    elif mode == "Temperature Sweep":
        st.subheader("Temperature Sweep Graphs")
        grid_on = st.radio("Grid lines:", ["On","Off"], horizontal=True) == "On"

        st.markdown("**Select mixes to plot**")
        mixes = []
        for name in sorted(processed):
            label = name.rsplit('.',1)[0]
            if st.checkbox(label, value=True, key=f"cb_graph_{name}"):
                mixes.append(name)
        if not mixes:
            st.info("Select at least one mix to see the panels.")
            st.stop()

        # grab & sort the three dyn-F values (as floats)
        dynFs = sorted(processed[mixes[0]].keys())

        metrics = ["G'", "G''", "G*", "Tan Delta"]
        for metric in metrics:
            st.markdown(f"##### {metric} vs Temperature")
            cols = st.columns(len(dynFs), gap="large")
            for dynF, col in zip(dynFs, cols):
                with col:
                    # clean label (no markdown) so the '*' won’t mangle
                    label = f"{metric} vs Temperature at {int(dynF)} N"
                    key   = f"title_{metric}_{int(dynF)}"
                    title = st.text_input("Rename Plot Title below", value=label, key=key)

                    fig, ax = plt.subplots()
                    for name in mixes:
                        sub = processed[name][dynF]  # already a copy
                        # aggregate by integer °C
                        deg = sub['Temperature (°C)'].round().astype(int)
                        col_raw = f"{metric}_smooth" if metric!="G*" else "G*_smooth"
                        agg = (sub.assign(deg=deg)
                                .groupby('deg')[col_raw]
                                .mean()
                                .reset_index()
                                .rename(columns={'deg': 'Temperature (°C)', col_raw:'y'}))
                        # second smooth + edge‐fill
                        agg['y_smooth'] = agg['y'].rolling(window=20, center=True).mean()
                        first = agg['y_smooth'].first_valid_index()
                        if first and first>0:     agg.loc[:first-1,'y_smooth'] = agg.loc[:first-1,'y']
                        last  = agg['y_smooth'].last_valid_index()
                        if last and last < len(agg)-1: agg.loc[last+1:,'y_smooth'] = agg.loc[last+1:,'y']

                        x = agg['Temperature (°C)']
                        y = agg['y_smooth']
                        if metric in ("G'","G''","G*"):
                            y = y * 1e-6

                        ax.plot(x, y, label=name.rsplit('.',1)[0], linewidth=1.5)

                    ax.set_xlim(agg['Temperature (°C)'].min(), agg['Temperature (°C)'].max())
                    ax.set_ylim(bottom=0)
                    ax.margins(y=0)
                    ax.grid(grid_on)
                    ax.set_xlabel("Temperature (°C)")
                    unit = " [MPa]" if metric in ("G'","G''","G*") else ""
                    ax.set_ylabel(metric + unit)
                    ax.set_title(title)
                    ax.legend(fontsize="small", loc="upper right")

                    col.pyplot(fig, use_container_width=True)




# — Key Values Interface —
with tab_key:
    st.subheader("Key Values")

    if mode=="Strain Sweep":
        key_names=[
            "DEFORMATION G\" MAX (%) Go","DEFORMATION G\" MAX (%) Return",
            "G' 10% Return (MPa)","G' 35% Return (MPa)",
            "G' DEF MIN Go (%)","G' DEF MIN Return (%)","G' DEF MAX Go (%)","G' DEF MAX Return (%)",
            "G'' 10% Go (MPa)","G'' 10% Return (MPa)","G'' 20% Return (MPa)","G'' 35% Return (MPa)",
            "G'' MAX Return (MPa)","G'' DEF MIN Return (MPa)",
            "G* 0.2% Go (MPa)","G* 0.2% Return (MPa)","G* 0.6% Return (MPa)",
            "G* 1% Go (MPa)","G* 1% Return (MPa)","G* 2% Go (MPa)","G* 2% Return (MPa)",
            "G* 10% Go (MPa)","G* 10% Return (MPa)","G* 15% Go (MPa)","G* 15% Return (MPa)",
            "G* 20% Go (MPa)","G* 20% Return (MPa)","G* 25% Go (MPa)","G* 25% Return (MPa)",
            "G* 50% Go (MPa)","G* 50% Return (MPa)","G* 97% Go (MPa)","G* 97% Return (MPa)",
            "G* DEF MIN Go (MPa)","G* DEF MIN Return (MPa)","G* DEF MAX Go (MPa)","G* DEF MAX Return (MPa)",
            "Tan D - 0.1% Return","Tan D - 1% Go","Tan D - 1% Return","Tan D - 10% Go","Tan D - 10% Return",
            "Tan D - 20% Go","Tan D - 20% Return","Tan D MAX Return","Tan D DEF MAX Return","Tan D DEF MIN Return","Tan D MAX Go"
        ]
        cols=[n.rsplit('.',1)[0] for n in sorted(processed)]
        summary_df=pd.DataFrame(index=key_names,columns=cols)
        for name,df in processed.items():
            mix=name.rsplit('.',1)[0]
            peak=df["dyn Str"].idxmax();df_go,df_ret=df.loc[:peak],df.loc[peak:]
            # i=df_go["G''"].idxmax();      summary_df.at["DEFORMATION G\" MAX (%) Go",mix]=df_go.loc[i,"dyn Str"]*100
            # i=df_ret["G''"].idxmax();     summary_df.at["DEFORMATION G\" MAX (%) Return",mix]=df_ret.loc[i,"dyn Str"]*100
            # G' value where dynamic Strain is closest to 0.10 and 0.35
            i=(df_ret["dyn Str"]-0.05).abs().idxmin(); summary_df.at["G' 10% Return (MPa)",mix]=df_ret.loc[i,"G'"]*1e-6
            i=(df_ret["dyn Str"]-0.175).abs().idxmin(); summary_df.at["G' 35% Return (MPa)",mix]=df_ret.loc[i,"G'"]*1e-6
            # dynamic Strain at G' minimum and maximum
            # i=df_go["G' (Pa)"].idxmin();       summary_df.at["G' DEF MIN Go (%)",mix]=df_go.loc[i,"dyn Str"]*100
            # i=df_ret["G' (Pa)"].idxmin();      summary_df.at["G' DEF MIN Return (%)",mix]=df_ret.loc[i,"dyn Str"]*100
            # i=df_go["G' (Pa)"].idxmax();       summary_df.at["G' DEF MAX Go (%)",mix]=df_go.loc[i,"dyn Str"]*100
            # i=df_ret["G' (Pa)"].idxmax();      summary_df.at["G' DEF MAX Return (%)",mix]=df_ret.loc[i,"dyn Str"]*100
            # G'' value where dynamic Strain closest to 0.10, 0.20 and 0.35
            i=(df_go["dyn Str"]-0.05).abs().idxmin();  summary_df.at["G'' 10% Go (MPa)",mix]=df_go.loc[i,"G''"]*1e-6
            i=(df_ret["dyn Str"]-0.05).abs().idxmin(); summary_df.at["G'' 10% Return (MPa)",mix]=df_ret.loc[i,"G''"]*1e-6
            i=(df_ret["dyn Str"]-0.10).abs().idxmin(); summary_df.at["G'' 20% Return (MPa)",mix]=df_ret.loc[i,"G''"]*1e-6
            i=(df_ret["dyn Str"]-0.175).abs().idxmin(); summary_df.at["G'' 35% Return (MPa)",mix]=df_ret.loc[i,"G''"]*1e-6
            summary_df.at["G'' MAX Return (MPa)",mix]=df_ret["G''"].max()*1e-6
            # dynamic Strain at G'' minimum
            # i=df_ret["G''"].idxmin();      summary_df.at["G'' DEF MIN Return (MPa)",mix]=df_ret.loc[i,"dyn Str"]
            # G* values at specified strains
            pairs=[(0.001,"G* 0.2% Go (MPa)",df_go),(0.001,"G* 0.2% Return (MPa)",df_ret),
                (0.003,"G* 0.6% Return (MPa)",df_ret),(0.005,"G* 1% Go (MPa)",df_go),
                (0.005,"G* 1% Return (MPa)",df_ret),(0.01,"G* 2% Go (MPa)",df_go),
                (0.01,"G* 2% Return (MPa)",df_ret),(0.05,"G* 10% Go (MPa)",df_go),
                (0.05,"G* 10% Return (MPa)",df_ret),(0.075,"G* 15% Go (MPa)",df_go),
                (0.075,"G* 15% Return (MPa)",df_ret),(0.1,"G* 20% Go (MPa)",df_go),
                (0.1,"G* 20% Return (MPa)",df_ret),(0.125,"G* 25% Go (MPa)",df_go),
                (0.125,"G* 25% Return (MPa)",df_ret),(0.25,"G* 50% Go (MPa)",df_go),
                (0.25,"G* 50% Return (MPa)",df_ret),(0.485,"G* 97% Go (MPa)",df_go),
                (0.485,"G* 97% Return (MPa)",df_ret)]
            for T,label,phase_df in pairs:
                if phase_df.empty: summary_df.at[label,mix]="N/A"
                else:
                    idx=(phase_df["dyn Str"]-T).abs().idxmin()
                    summary_df.at[label,mix]=phase_df.loc[idx,"G*"]*1e-6
            # G* def min/max
            # summary_df.at["G* DEF MIN Go (MPa)",mix]=df_go["G*"].min()*1e-6 if not df_go.empty else "N/A"
            # summary_df.at["G* DEF MIN Return (MPa)",mix]=df_ret["G*"].min()*1e-6 if not df_ret.empty else "N/A"
            # summary_df.at["G* DEF MAX Go (MPa)",mix]=df_go["G*"].max()*1e-6 if not df_go.empty else "N/A"
            # summary_df.at["G* DEF MAX Return (MPa)",mix]=df_ret["G*"].max()*1e-6 if not df_ret.empty else "N/A"
            # Tan D at specified strains and extremes
            i=(df_ret["dyn Str"]-0.0005).abs().idxmin(); summary_df.at["Tan D - 0.1% Return",mix]=df_ret.loc[i,"Tan Delta"]
            i=(df_go["dyn Str"]-0.005).abs().idxmin();  summary_df.at["Tan D - 1% Go",mix]=df_go.loc[i,"Tan Delta"]
            i=(df_ret["dyn Str"]-0.005).abs().idxmin(); summary_df.at["Tan D - 1% Return",mix]=df_ret.loc[i,"Tan Delta"]
            i=(df_go["dyn Str"]-0.05).abs().idxmin();   summary_df.at["Tan D - 10% Go",mix]=df_go.loc[i,"Tan Delta"]
            i=(df_ret["dyn Str"]-0.05).abs().idxmin();  summary_df.at["Tan D - 10% Return",mix]=df_ret.loc[i,"Tan Delta"]
            i=(df_go["dyn Str"]-0.1).abs().idxmin();   summary_df.at["Tan D - 20% Go",mix]=df_go.loc[i,"Tan Delta"]
            i=(df_ret["dyn Str"]-0.1).abs().idxmin();  summary_df.at["Tan D - 20% Return",mix]=df_ret.loc[i,"Tan Delta"]
            summary_df.at["Tan D MAX Return",mix]=df_ret["Tan Delta"].max()
            # i=df_ret["Tan Delta"].idxmax();         summary_df.at["Tan D DEF MAX Return",mix]=df_ret.loc[i,"dyn Str"]*100
            # i=df_ret["Tan Delta"].idxmin();         summary_df.at["Tan D DEF MIN Return",mix]=df_ret.loc[i,"dyn Str"]*100
            summary_df.at["Tan D MAX Go",mix]=df_go["Tan Delta"].max()
        st.dataframe(summary_df,use_container_width=True)



    elif mode=="Temperature Sweep":

        # 1) The rows we want in every mini‐test table:
        key_names = [
            "G'' 10°C (MPa)","G'' 90°C (MPa)","G'' MAX (MPa)",
            "G* -30°C (MPa)","G* -20°C (MPa)","G* -10°C (MPa)","G* 0°C (MPa)","G* 10°C (MPa)",
            "G* 20°C (MPa)","G* 30°C (MPa)","G* 40°C (MPa)","G* 50°C (MPa)","G* 60°C (MPa)",
            "G* 90°C (MPa)","G* 100°C (MPa)","G\"/G*² -20°C","G\"/G*² MAX",
            "Slope (G*98°C-G*75°C)/ΔT","T (°C) G'' MAX","T (°C) G''/G*² MAX","T (°C) Tan D MAX",
            "Tan -30°C","Tan -20°C","Tan -10°C","Tan 0°C","Tan 10°C","Tan 20°C",
            "Tan 30°C","Tan 40°C","Tan 50°C","Tan 60°C","Tan 90°C","Tan 100°C",
            "Tan MAX","Tan Elastomer (°C)",
            "Temp (°C) G*=1.5MPa","Temp (°C) G*=3MPa","Temp (°C) G*=5MPa",
            "Temp (°C) G*=10MPa","Temp (°C) G*=100MPa"
        ]

        # 2) Gather ALL dyn F values across mixes:
        all_dynFs = sorted({df_key for mix_dict in processed.values() for df_key in mix_dict.keys()}, key=float)

        # 3) For each mini‐test (dyn F), build one DataFrame whose columns are the mixes:
        for dynF in all_dynFs:
            
            # one column per mix
            mix_names = [name.rsplit('.',1)[0] for name in sorted(processed)]
            summary_df = pd.DataFrame(index=key_names, columns=mix_names)

            for full_name, mix_dict in processed.items():
                mix = full_name.rsplit('.',1)[0]
                sub = mix_dict[dynF]  # this DataFrame for that mini‐test

                # nearest‐temp picks for G'' @ 10 & 90
                for T in (10, 90):
                    idx = (sub["Temperature (°C)"] - T).abs().idxmin()
                    summary_df.at[f"G'' {T}°C (MPa)", mix] = sub.at[idx, "G''"] * 1e-6
                summary_df.at["G'' MAX (MPa)", mix] = sub["G''"].max() * 1e-6

                # G* & Tan at each target temperature
                temps = (-30,-20,-10,0,10,20,30,40,50,60,90,100)
                for T in temps:
                    idx = (sub["Temperature (°C)"] - T).abs().idxmin()
                    summary_df.at[f"G* {T}°C (MPa)", mix] = sub.at[idx, "G*"] * 1e-6
                    summary_df.at[f"Tan {T}°C",       mix] = sub.at[idx, "Tan Delta"]

                # G''/G*² at -20°C and global max
                idx20 = (sub["Temperature (°C)"] + 20).abs().idxmin()
                val20 = sub.at[idx20, "G''"] / (sub.at[idx20, "G*"]**2)
                summary_df.at["G\"/G*² -20°C", mix] = val20
                summary_df.at["G\"/G*² MAX",     mix] = (sub["G''"]/(sub["G*"]**2)).max()

                # slope (98 vs 75)
                i98 = (sub["Temperature (°C)"] - 98).abs().idxmin()
                i75 = (sub["Temperature (°C)"] - 75).abs().idxmin()
                slope = ((sub.at[i98,"G*"] - sub.at[i75,"G*"]) * 1e-6) / (98 - 75)
                summary_df.at["Slope (G*98°C-G*75°C)/ΔT", mix] = slope

                # temperatures of maxima
                iGpp = sub["G''"].idxmax()
                summary_df.at["T (°C) G'' MAX", mix]     = sub.at[iGpp, "Temperature (°C)"]
                ratio = sub["G''"]/(sub["G*"]**2)
                iR   = ratio.idxmax()
                summary_df.at["T (°C) G''/G*² MAX", mix] = sub.at[iR, "Temperature (°C)"]
                iTan = sub["Tan Delta"].idxmax()
                summary_df.at["T (°C) Tan D MAX", mix]   = sub.at[iTan, "Temperature (°C)"]

                # Tan maxima & gel point
                summary_df.at["Tan MAX", mix]            = sub["Tan Delta"].max()
                iGel = sub["Tan Delta"].idxmax()
                summary_df.at["Tan Elastomer (°C)", mix] = sub.at[iGel, "Temperature (°C)"]

                # G* threshold matches ±5%
                for thr in (1.5, 3, 5, 10, 100):
                    tol = thr * 0.05
                    diff = (sub["G*"]*1e-6 - thr).abs()
                    matches = diff <= tol
                    if matches.any():
                        idx0 = matches.idxmax()
                        summary_df.at[f"Temp (°C) G*={thr}MPa", mix] = sub.at[idx0, "Temperature (°C)"]
                    else:
                        summary_df.at[f"Temp (°C) G*={thr}MPa", mix] = np.nan

            # 4) Offer one Download button + show the table
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="Key Values")
            buf.seek(0)
            c1, c2 = st.columns([9,1])
            with c1:
                label = f"{int(float(dynF))} N"
                st.markdown(f"#### dyn F = {label}")
            with c2:
                st.download_button(label=f"Download {label}", data=buf, file_name=f"{label}_keys.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.dataframe(summary_df, use_container_width=True, height=400)
            st.markdown("---")





# — Data Interface —
with tab_data:
    st.subheader("Raw Data Tables")

    if mode == "Strain Sweep":
        for name in mixes:
            st.markdown(f"**{name.rsplit('.',1)[0]}**")
            df = processed[name]
            # drop all the '_smooth' cols
            drop_cols = [f"{m}_smooth" for m in ["G'", "G''", "Tan Delta", "G*"]]
            df_display = df.drop(columns=drop_cols).copy()
            # find G-columns in Pa, convert → MPa
            g_cols = [c for c in df_display.columns if "(Pa)" in c]
            for c in g_cols:
                df_display[c] = pd.to_numeric(df_display[c], errors='coerce') / 1e6
            df_display = df_display.rename(columns={c: c.replace("(Pa)", "(MPa)") for c in g_cols})
            st.dataframe(df_display, use_container_width=True)

    elif mode == "Temperature Sweep":
        import io

        for name in mixes:
            

            # Prepare one Excel workbook per mix
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                for dynF, df in processed[name].items():
                    # drop the _smooth cols
                    drop_cols = [f"{m}_smooth" for m in ["G'", "G''", "Tan Delta", "G*"]]
                    df_disp = df.drop(columns=drop_cols).copy()

                    # convert G cols to MPa
                    for c in ["G'", "G''", "G*"]:
                        if c in df_disp:
                            df_disp[c] = pd.to_numeric(df_disp[c], errors='coerce') / 1e6

                    # rename for clarity
                    df_disp = df_disp.rename(columns={
                        "G'": "G' (MPa)",
                        "G''": "G'' (MPa)",
                        "G*": "G* (MPa)"
                    })

                    # move "G* (MPa)" into position 5
                    cols = list(df_disp.columns)
                    if "G* (MPa)" in cols:
                        cols.insert(5, cols.pop(cols.index("G* (MPa)")))
                        df_disp = df_disp[cols]

                    # write sheet named by dyn F
                    sheet = f"{int(float(dynF))}N"
                    df_disp.to_excel(writer, sheet_name=sheet, index=False)

            # finalize buffer
            buffer.seek(0)

            # download button + optional preview
            c1, c2 = st.columns([8.5,1.5])
            with c1:
                mix = name.rsplit('.',1)[0]
                st.markdown(f"##### {mix}")
                st.markdown("The data will be downloaded as an Excel file with separate sheets for each dyn F value.")
            with c2:
                st.download_button(
                    label="Download all data",
                    data=buffer,
                    file_name=f"{mix}_raw_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
                        # (Optional) show first mini-test sheet as preview
            first_df = next(iter(processed[name].values())).drop(columns=drop_cols).copy()
            for c in ["G'", "G''", "G*"]:
                if c in first_df:
                    first_df[c] = pd.to_numeric(first_df[c], errors='coerce') / 1e6
            first_df = first_df.rename(columns={"G'":"G' (MPa)","G''":"G'' (MPa)","G*":"G* (MPa)"})
            cols = list(first_df.columns)
            if "G* (MPa)" in cols:
                cols.insert(5, cols.pop(cols.index("G* (MPa)")))
                first_df = first_df[cols]
            st.dataframe(first_df, use_container_width=True, height=300)


