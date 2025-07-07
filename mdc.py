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
    raw = buffer.readlines() if hasattr(buffer, "readlines") else open(buffer, 'rb').readlines()
    lines = [
        L.decode('latin-1', errors='replace') if isinstance(L, (bytes, bytearray)) else L
        for L in raw
    ]

    new_cols = ['dyn F (N)', 'Freq (Hz)', 'Time (s)', "G'", "G''", 'Tan Delta', 'Temperature (°C)', 'dyn Str', 'Time1']

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
    metrics = ["G'", "G''", "Tan Delta"]

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors='coerce')
        df[f"{m}_smooth"] = df[m].rolling(window=2, center=True).mean()

    # compute G* = sqrt(G'^2 + G''^2) and then smooth it
    gstar = np.sqrt(df["G'"]**2 + df["G''"]**2)
    df.insert(loc=5, column="G*", value=gstar)
    df["G*_smooth"] = df["G*"].rolling(window=2, center=True).mean()

    return df


@st.cache_data
def aggregate_by_degree(df: pd.DataFrame, degree_col: str, smooth_cols: list[str]) -> pd.DataFrame:
    # make a copy and round the degree column to ints
    df_ = df.copy()
    df_['deg'] = df_[degree_col].round().astype(int)
    # group by that integer degree, average all the smoothed metrics
    agg = (
        df_
        .groupby('deg')[smooth_cols]
        .mean()
        .reset_index()
        .rename(columns={'deg': degree_col})
    )
    return agg


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

if mode=="Temperature Sweep":
    raw_force_aggs = {name: df.assign(deg=df['Temperature (°C)'].round().astype(int), dynF=pd.to_numeric(df['dyn F (N)'],errors='coerce')).groupby('deg')['dynF'].mean() for name,df in processed.items()}
    raw_val_aggs   = {name: df.assign(deg=df['Temperature (°C)'].round().astype(int)).groupby('deg')[["G''","G*","Tan Delta"]].mean() for name,df in processed.items()}



tab_graph, tab_key, tab_data = st.tabs(["Graph Interface","Key Values","Data Interface"])



# — Graph Interface —
with tab_graph:

    if mode == "Strain Sweep":
        st.subheader("Strain Sweep Graphs")
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            phase = st.radio("Phase", ["Both","Go","Return"], horizontal=True)
        with col2:
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

        metrics = ["G'", "G''", "G*", "Tan Delta"]
        # render in 2×2 grid, each with its own title input
        for row in (metrics[:2], metrics[2:]):
            plot_cols = st.columns(2, gap="large")
            for metric, pc in zip(row, plot_cols):
                with pc:
                    title = st.text_input(f"**Title for {metric}**", value=f"{metric} vs Strain ({phase})", key=f"title_{metric}")
                    fig, ax = plt.subplots()
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
                            y *= 1e-6
                        ax.plot(df["dyn Str"], y, label=name.rsplit('.',1)[0], linewidth=1.5)

                    # force exponent range from 10^-3 upward
                    min_e = -3
                    max_e = int(np.ceil(np.log10(df["dyn Str"].max())))
                    ticks = [10**e for e in range(min_e, max_e+1)]

                    ax.set_xscale('log'); ax.set_xticks(ticks); ax.set_xlim(ticks[0], ticks[-1]); ax.set_xticklabels([f"{t:g}" for t in ticks])
                    ax.set_ylim(bottom=0); ax.margins(y=0); ax.grid(grid_on); ax.set_xlabel("Strain [%]")
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

        metrics = ["G'", "G''", "G*", "Tan Delta"]
        for row in (metrics[:2], metrics[2:]):
            plot_cols = st.columns(2, gap="large")
            for metric, pc in zip(row, plot_cols):
                with pc:
                    title = st.text_input(f"**Title for {metric}**", value=f"{metric} vs Temperature (°C)", key=f"title_temp_{metric}")
                    fig, ax = plt.subplots()
                    for name in mixes:
                        df = processed[name].copy()
                        deg = df["Temperature (°C)"].round().astype(int)
                        y = df[f"{metric}_smooth"] if metric != "G*" else df["G*_smooth"]
                        agg = (
                            pd.DataFrame({ "Temperature (°C)": deg, "y": y })
                            .groupby("Temperature (°C)")["y"]
                            .mean()
                            .reset_index()
                        )
                        agg[f"{metric}_agg_smooth"] = agg["y"].rolling(window=20, center=True).mean()
                        x_vals = agg["Temperature (°C)"]
                        y_vals = agg[f"{metric}_agg_smooth"]
                        if metric in ("G'", "G''", "G*"):
                            y_vals *= 1e-6
                        ax.plot(x_vals, y_vals,
                                label=name.rsplit('.',1)[0],
                                linewidth=1.5)

                    ax.set_xlim(df["Temperature (°C)"].min(), df["Temperature (°C)"].max())
                    ax.set_ylim(bottom=0)
                    ax.margins(y=0)
                    ax.grid(grid_on)
                    ax.set_xlabel("Temperature (°C)")
                    unit = " [MPa]" if metric in ("G'", "G''", "G*") else ""
                    ax.set_ylabel(metric + unit)
                    ax.set_title(title)
                    ax.legend(fontsize="small", loc="upper right")
                    pc.pyplot(fig, use_container_width=True)    




# — Key Values Interface —
with tab_key:
    st.subheader("Key Values")
    if mode=="Temperature Sweep":
        key_names = [
            "Def % -20°C","Def % -10°C","Def % 10°C","Def % 30°C","Def % 60°C","Def % 100°C",
            "G'' 10°C (MPa)","G'' 90°C (MPa)","G'' MAX (MPa)",
            "G* -30°C (MPa)","G* -20°C (MPa)","G* -10°C (MPa)","G* 0°C (MPa)","G* 10°C (MPa)",
            "G* 20°C (MPa)","G* 30°C (MPa)","G* 40°C (MPa)","G* 50°C (MPa)","G* 60°C (MPa)",
            "G* 90°C (MPa)","G* 100°C (MPa)","G\"/G*² -20°C","G\"/G*² MAX",
            "Slope (G*98°C-G*75°C)/ΔT","T (°C) G'' MAX","T (°C) G''/G*² MAX","T (°C) Tan D MAX",
            "Tan -30°C","Tan -20°C","Tan -10°C","Tan 0°C","Tan 10°C","Tan 20°C",
            "Tan 30°C","Tan 40°C","Tan 50°C","Tan 60°C","Tan 90°C","Tan 100°C",
            "Tan MAX","Tan Elastomer (°C)",
            "Temp (°C) G*=1.5MPa","Temp (°C) G*=3MPa","Temp (°C) G*=5MPa","Temp (°C) G*=10MPa","Temp (°C) G*=100MPa"
        ]
        cols       = [n.rsplit('.',1)[0] for n in sorted(processed)]
        summary_df = pd.DataFrame(index=key_names, columns=cols)
        thresholds = [1.5,3,5,10,100]
        for name,df in processed.items():
            mix        = name.rsplit('.',1)[0]
            f          = raw_force_aggs[name]
            v          = raw_val_aggs[name]
            summary_df.at["Def % -20°C",      mix] = f.get(-20,   np.nan)
            summary_df.at["Def % -10°C",      mix] = f.get(-10,   np.nan)
            summary_df.at["Def % 10°C",       mix] = f.get( 10,   np.nan)
            summary_df.at["Def % 30°C",       mix] = f.get( 30,   np.nan)
            summary_df.at["Def % 60°C",       mix] = f.get( 60,   np.nan)
            summary_df.at["Def % 100°C",      mix] = f.get(100,   np.nan)
            summary_df.at["G'' 10°C (MPa)",   mix] = v.at[ 10,"G''"]*1e-6
            summary_df.at["G'' 90°C (MPa)",   mix] = v.at[ 90,"G''"]*1e-6
            summary_df.at["G'' MAX (MPa)",    mix] = df["G''"].max()*1e-6
            for T in (-30,-20,-10,0,10,20,30,40,50,60,90,100):
                summary_df.at[f"G* {T}°C (MPa)",mix] = v.at[T,"G*"]*1e-6
                summary_df.at[f"Tan {T}°C",       mix] = raw_val_aggs[name].at[T,"Tan Delta"]
            summary_df.at["G\"/G*² -20°C",     mix] = v.at[-20,"G''"]/(v.at[-20,"G*"]**2)
            summary_df.at["G\"/G*² MAX",       mix] = df["G''"].max()/(df["G*"].max()**2)
            summary_df.at["Tan MAX",           mix] = df["Tan Delta"].max()
            summary_df.at["Tan Elastomer (°C)",mix] = df.loc[df["Tan Delta"].idxmax(),"Temperature (°C)"]
            summary_df.at["Slope (G*98°C-G*75°C)/ΔT",mix] = ((v.at[98,"G*"]-v.at[75,"G*"])*1e-6)/23
            summary_df.at["T (°C) G'' MAX",    mix] = df.loc[df["G''"].idxmax(),"Temperature (°C)"]
            ratio = df["G''"]/(df["G*"]**2)
            summary_df.at["T (°C) G''/G*² MAX",mix] = df.loc[ratio.idxmax(),"Temperature (°C)"]
            summary_df.at["T (°C) Tan D MAX",  mix] = df.loc[df["Tan Delta"].idxmax(),"Temperature (°C)"]
            gstar_mp = df["G*_smooth"] * 1e-6
            tol      = 2.0
            for thr in thresholds:
                mask    = (gstar_mp >= thr - tol) & (gstar_mp <= thr + tol)
                temp_val= df.loc[mask, "Temperature (°C)"].iloc[0] if mask.any() else np.nan
                summary_df.at[f"Temp (°C) G*={thr}MPa", mix] = temp_val

        st.dataframe(summary_df, use_container_width=True, height=600)
    

    elif mode=="Strain Sweep":
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
            i=df_go["G''"].idxmax();      summary_df.at["DEFORMATION G\" MAX (%) Go",mix]=df_go.loc[i,"dyn Str"]*100
            i=df_ret["G''"].idxmax();     summary_df.at["DEFORMATION G\" MAX (%) Return",mix]=df_ret.loc[i,"dyn Str"]*100
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
            i=df_ret["G''"].idxmin();      summary_df.at["G'' DEF MIN Return (MPa)",mix]=df_ret.loc[i,"dyn Str"]
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
            summary_df.at["G* DEF MIN Go (MPa)",mix]=df_go["G*"].min()*1e-6 if not df_go.empty else "N/A"
            summary_df.at["G* DEF MIN Return (MPa)",mix]=df_ret["G*"].min()*1e-6 if not df_ret.empty else "N/A"
            summary_df.at["G* DEF MAX Go (MPa)",mix]=df_go["G*"].max()*1e-6 if not df_go.empty else "N/A"
            summary_df.at["G* DEF MAX Return (MPa)",mix]=df_ret["G*"].max()*1e-6 if not df_ret.empty else "N/A"
            # Tan D at specified strains and extremes
            i=(df_ret["dyn Str"]-0.0005).abs().idxmin(); summary_df.at["Tan D - 0.1% Return",mix]=df_ret.loc[i,"Tan Delta"]
            i=(df_go["dyn Str"]-0.005).abs().idxmin();  summary_df.at["Tan D - 1% Go",mix]=df_go.loc[i,"Tan Delta"]
            i=(df_ret["dyn Str"]-0.005).abs().idxmin(); summary_df.at["Tan D - 1% Return",mix]=df_ret.loc[i,"Tan Delta"]
            i=(df_go["dyn Str"]-0.05).abs().idxmin();   summary_df.at["Tan D - 10% Go",mix]=df_go.loc[i,"Tan Delta"]
            i=(df_ret["dyn Str"]-0.05).abs().idxmin();  summary_df.at["Tan D - 10% Return",mix]=df_ret.loc[i,"Tan Delta"]
            i=(df_go["dyn Str"]-0.1).abs().idxmin();   summary_df.at["Tan D - 20% Go",mix]=df_go.loc[i,"Tan Delta"]
            i=(df_ret["dyn Str"]-0.1).abs().idxmin();  summary_df.at["Tan D - 20% Return",mix]=df_ret.loc[i,"Tan Delta"]
            summary_df.at["Tan D MAX Return",mix]=df_ret["Tan Delta"].max()
            i=df_ret["Tan Delta"].idxmax();         summary_df.at["Tan D DEF MAX Return",mix]=df_ret.loc[i,"dyn Str"]*100
            i=df_ret["Tan Delta"].idxmin();         summary_df.at["Tan D DEF MIN Return",mix]=df_ret.loc[i,"dyn Str"]*100
            summary_df.at["Tan D MAX Go",mix]=df_go["Tan Delta"].max()
        st.dataframe(summary_df,use_container_width=True)


    else:
        st.info("Key values for Strain Sweep coming soon.")





# — Data Interface —
with tab_data:
    st.subheader("Raw Data Tables")
    for name in mixes:
        st.markdown(f"**{name.rsplit('.',1)[0]}**")
        df = processed[name]

        # drop the “_smooth” columns as before
        drop_cols = [f"{m}_smooth" for m in ["G'","G''","Tan Delta","G*"]]
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

