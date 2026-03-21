
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="FY26 vs FY25 Dashboard", layout="wide")

MONTH_ORDER = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
FY_ORDER = ['FY24','FY25','FY26','FY27']

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        default_path = Path(__file__).with_name("monthly_sellin.csv")
        df = pd.read_csv(default_path)
    required = {"brand","fiscal_year","month","volume_9lc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    df["fiscal_year"] = pd.Categorical(df["fiscal_year"], categories=FY_ORDER, ordered=True)
    df["volume_9lc"] = pd.to_numeric(df["volume_9lc"], errors="coerce").fillna(0)
    return df.sort_values(["brand","fiscal_year","month"])

def fmt_int(x):
    return f"{int(round(x)):,}"

def mini_sparkline(values, width=90, height=22, color="#4f7dff", baseline=False):
    vals = list(values)
    if not vals:
        return ""
    vmin = min(vals); vmax = max(vals)
    if vmax == vmin:
        pts = [f"{i*(width/(len(vals)-1 if len(vals)>1 else 1)):.1f},{height/2:.1f}" for i,_ in enumerate(vals)]
        points = " ".join(pts)
    else:
        pts=[]
        for i,v in enumerate(vals):
            x = i*(width/(len(vals)-1 if len(vals)>1 else 1))
            y = height - ((v-vmin)/(vmax-vmin))*(height-4) - 2
            pts.append(f"{x:.1f},{y:.1f}")
        points = " ".join(pts)
    base = f'<line x1="0" y1="{height-1}" x2="{width}" y2="{height-1}" stroke="#ddd" stroke-width="1"/>' if baseline else ""
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{base}<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" /></svg>'

def mini_var_bars(values, width=90, height=24):
    vals = list(values)
    if not vals:
        return ""
    max_abs = max(abs(v) for v in vals) or 1
    n = len(vals); gap = 2
    bar_w = max(3, (width-(n-1)*gap)//n)
    zero = height/2
    rects = []
    for i,v in enumerate(vals):
        x = i*(bar_w+gap)
        bar_h = abs(v)/max_abs*(height/2-2)
        y = zero-bar_h if v>=0 else zero
        color = "#4fa3ff" if v>=0 else "#ff9a9a"
        rects.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" rx="1" />')
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><line x1="0" y1="{zero:.1f}" x2="{width}" y2="{zero:.1f}" stroke="#ddd" stroke-width="1"/>' + "".join(rects) + '</svg>'

def build_table_html(pivot):
    total_selected = pivot["FY26"].sum() or 1
    rows=[]
    for _, r in pivot.iterrows():
        monthly = r[[f"FY26_{m}" for m in MONTH_ORDER]].tolist()
        monthly_compare = r[[f"FY25_{m}" for m in MONTH_ORDER]].tolist()
        monthly_var = [a-b for a,b in zip(monthly, monthly_compare)]
        diff = r["FY26"] - r["FY25"]
        pct = 0 if r["FY25"] == 0 else diff / r["FY25"] * 100
        share = r["FY26"] / total_selected * 100
        diff_color = "#0f9d58" if diff > 0 else "#d93025" if diff < 0 else "#666"
        bar_width = min(27, abs(diff)/30000*27)
        left_pos = 27 if diff >= 0 else max(0, 27 - bar_width)
        diff_bar = (
            f"<div style='display:flex; align-items:center; gap:6px;'>"
            f"<div style='width:52px; text-align:right; color:{diff_color}; font-weight:700;'>{diff:,.0f}</div>"
            f"<div style='background:#efefef; width:54px; height:12px; position:relative; border-radius:2px;'>"
            f"<div style='position:absolute; left:{left_pos}px; width:{bar_width:.1f}px; height:12px; background:{'#69c36d' if diff>=0 else '#e35d52'}; border-radius:2px;'></div>"
            f"</div></div>"
        )
        rows.append(
            "<tr>"
            f"<td class='brand'>{r['brand']}</td>"
            f"<td class='num'>{fmt_int(r['FY24'])}</td>"
            f"<td class='num fy25'>{fmt_int(r['FY25'])}</td>"
            f"<td class='num fy26'>{fmt_int(r['FY26'])}</td>"
            f"<td class='num'>{fmt_int(r['FY27'])}</td>"
            f"<td class='num'>{share:.1f}%</td>"
            f"<td>{mini_sparkline(monthly, color='#4f7dff')}</td>"
            f"<td>{diff_bar}</td>"
            f"<td class='num' style='color:{diff_color};font-weight:700;'>{pct:.0f}%</td>"
            f"<td>{mini_var_bars(monthly_var)}</td>"
            "</tr>"
        )
    total_diff = pivot["FY26"].sum() - pivot["FY25"].sum()
    total_pct = 0 if pivot["FY25"].sum() == 0 else total_diff / pivot["FY25"].sum() * 100
    total_row = (
        "<tr class='total-row'>"
        "<td class='brand'><b>Total</b></td>"
        f"<td class='num'><b>{fmt_int(pivot['FY24'].sum())}</b></td>"
        f"<td class='num'><b>{fmt_int(pivot['FY25'].sum())}</b></td>"
        f"<td class='num'><b>{fmt_int(pivot['FY26'].sum())}</b></td>"
        f"<td class='num'><b>{fmt_int(pivot['FY27'].sum())}</b></td>"
        "<td class='num'><b>100.0%</b></td><td></td>"
        f"<td class='num'><b>{fmt_int(total_diff)}</b></td>"
        f"<td class='num'><b>{total_pct:.0f}%</b></td><td></td></tr>"
    )
    style = '''
    <style>
      table.dashboard {width:100%; border-collapse:collapse; font-size:12px;}
      .dashboard th, .dashboard td {border-bottom:1px solid #e9e9e9; padding:6px 8px; vertical-align:middle;}
      .dashboard th {background:white; font-weight:700; text-align:left;}
      .dashboard td.num, .dashboard th.num {text-align:right; white-space:nowrap;}
      .dashboard td.fy25 {background:#ffb000; color:#111; font-weight:700;}
      .dashboard td.fy26 {background:#a8d8e8; color:#111; font-weight:700;}
      .dashboard tr.total-row td {background:#fafafa; border-top:2px solid #bbb;}
      .brand {white-space:nowrap;}
    </style>
    '''
    html = (
        style +
        "<table class='dashboard'><thead><tr>"
        "<th>brand</th><th class='num'>FY24 (9LC)</th><th class='num'>FY25 (9LC)</th>"
        "<th class='num'>FY26 (9LC)</th><th class='num'>FY27 (9LC)</th><th class='num'>share of FY26 Volume (%)</th>"
        "<th>Monthly Sell IN</th><th>FY26 vs FY25 (9LC)</th><th class='num'>FY26 vs FY25 (%)</th>"
        "<th>FY26 vs FY25 Monthly Variations</th></tr></thead><tbody>"
        + "".join(rows) + total_row + "</tbody></table>"
    )
    return html

st.title("FY26 vs FY25 Dashboard")
st.caption("Replica-style dashboard based on uploaded monthly sell-in CSV")

with st.sidebar:
    st.subheader("Controls")
    uploaded = st.file_uploader("Upload monthly_sellin.csv", type=["csv"])
    st.markdown("Expected columns: `brand`, `fiscal_year`, `month`, `volume_9lc`")
    show_top_n = st.slider("Brands shown", min_value=5, max_value=30, value=15, step=1)
    use_sample = st.checkbox("Use bundled sample data", value=uploaded is None)
    st.markdown("---")
    st.caption("Sample file is bundled in this folder.")

try:
    df = load_data(None if use_sample else uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

brands_available = sorted(df["brand"].unique().tolist())
with st.sidebar:
    selected_brands = st.multiselect("Brand filter", brands_available, default=brands_available)
if not selected_brands:
    st.warning("Select at least one brand.")
    st.stop()

fdf = df[df["brand"].isin(selected_brands)].copy()
top_brands = (
    fdf[fdf["fiscal_year"] == "FY26"]
    .groupby("brand", observed=True)["volume_9lc"].sum()
    .sort_values(ascending=False)
    .head(show_top_n)
    .index.tolist()
)
fdf = fdf[fdf["brand"].isin(top_brands)]

monthly = (
    fdf.groupby(["fiscal_year","month"], observed=True)["volume_9lc"]
    .sum()
    .reset_index()
    .sort_values(["fiscal_year","month"])
)

left, right = st.columns([1.0, 1.9], gap="large")

with left:
    monthly_p = monthly.pivot(index="month", columns="fiscal_year", values="volume_9lc").reindex(MONTH_ORDER)
    fig_line = go.Figure()
    colors = {"FY25":"#f6a000","FY26":"#a6d3e8"}
    for fy in ["FY25","FY26"]:
        fig_line.add_trace(go.Scatter(
            x=MONTH_ORDER, y=monthly_p[fy], mode="lines+markers", name=fy,
            line=dict(color=colors[fy], width=2), marker=dict(size=5)
        ))
    fig_line.update_layout(
        title="Monthly Sell IN : FY26 vs FY25", title_x=0.05, height=300,
        margin=dict(l=20,r=10,t=55,b=10), xaxis_title=None, yaxis_title="in 9LC units",
        legend=dict(orientation="h", y=-0.18)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    var = (monthly_p["FY26"] - monthly_p["FY25"]).fillna(0)
    colors_bar = ["#31c43f" if v > 0 else "#df2c1d" if v < 0 else "#999999" for v in var]
    fig_bar = go.Figure(go.Bar(x=MONTH_ORDER, y=var, marker_color=colors_bar,
                               text=[f"{int(v):,}" for v in var], textposition="inside"))
    fig_bar.update_layout(
        title="Month Sell IN Variations : FY26 vs FY25", title_x=0.05, height=300,
        margin=dict(l=20,r=10,t=55,b=10), xaxis_title=None, yaxis_title="in 9LC units",
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.markdown("##### FY26 vs FY25")
    st.caption("in 9LC cases")
    piv = (
        fdf.pivot_table(index="brand", columns="fiscal_year", values="volume_9lc", aggfunc="sum", fill_value=0, observed=True)
        .reindex(columns=FY_ORDER, fill_value=0)
        .reset_index()
    )
    monthly_wide = (
        fdf.pivot_table(index="brand", columns=["fiscal_year","month"], values="volume_9lc", aggfunc="sum", fill_value=0, observed=True)
    )
    monthly_wide.columns = [f"{a}_{b}" for a,b in monthly_wide.columns]
    table_df = piv.merge(monthly_wide.reset_index(), on="brand", how="left")
    table_df = table_df.sort_values("FY26", ascending=False).reset_index(drop=True)
    st.markdown(build_table_html(table_df), unsafe_allow_html=True)

    export_df = table_df[["brand","FY24","FY25","FY26","FY27"]].copy()
    export_df["FY26_vs_FY25_9LC"] = export_df["FY26"] - export_df["FY25"]
    export_df["FY26_vs_FY25_pct"] = np.where(export_df["FY25"] == 0, 0, (export_df["FY26"] - export_df["FY25"]) / export_df["FY25"])
    st.markdown("**Download table**")
    st.download_button("Download", data=export_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="fy26_vs_fy25_table.csv", mime="text/csv")
