import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Comparison between Sell IN RF Taiwan', layout='wide')

MONTH_ORDER = ['Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun']
PUBLISH_ORDER = ['RF1', 'RF2', 'RF3', 'RF4', 'RF5', 'RF6', 'RF7', 'RF8', 'RF9', 'RF10',
                 'RF11', 'RF12', 'RF1_FY26', 'RF2_FY26', 'RF3_FY26', 'RF4_FY26', 'RF5_FY26',
                 'RF6_FY26', 'RF7_FY26', 'RF8_FY26', 'RF9_FY26']
FY_ORDER = ['FY24', 'FY25', 'FY26', 'FY27']


def read_csv_any(uploaded):
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding='latin1')


def normalize_text_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def ensure_month_order(df, col='calendar_month_abb'):
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=MONTH_ORDER, ordered=True)
    return df


def safe_multiselect(label, options, default=None, key=None):
    options = [o for o in options if pd.notna(o)]
    default = default if default is not None else options
    return st.multiselect(label, options, default=default, key=key)


def kpi_block(title, value):
    st.markdown(f"<div style='padding:0.6rem 0.9rem;border:1px solid #ddd;border-radius:8px'>"
                f"<div style='font-size:0.9rem;color:#666'>{title}</div>"
                f"<div style='font-size:1.7rem;font-weight:700'>{value}</div></div>", unsafe_allow_html=True)


def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return ""


def fmt_pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return ""


def get_publish_series(df, publish_col):
    if publish_col not in df.columns:
        raise KeyError(f"Selected publish column '{publish_col}' is not present in Set_Up_All_RF_data.")
    s = pd.to_numeric(df[publish_col], errors='coerce').fillna(0)
    return s


def base_filtered_df(df, filters, publish_col, period_range=None, date_range=None):
    out = df.copy()
    for col, vals in filters.items():
        if col in out.columns and vals:
            out = out[out[col].isin(vals)]
    if period_range and 'fiscal_month' in out.columns:
        out = out[(pd.to_numeric(out['fiscal_month'], errors='coerce') >= period_range[0]) &
                  (pd.to_numeric(out['fiscal_month'], errors='coerce') <= period_range[1])]
    if date_range and 'period' in out.columns:
        out['period'] = pd.to_datetime(out['period'], errors='coerce')
        out = out[(out['period'] >= pd.to_datetime(date_range[0])) & (out['period'] <= pd.to_datetime(date_range[1]))]
    out = out[out['fiscal_year'].isin(FY_ORDER)] if 'fiscal_year' in out.columns else out
    out['Publish.Dimension'] = get_publish_series(out, publish_col)
    return out


def add_dimension(df, dimension_col):
    out = df.copy()
    if dimension_col not in out.columns:
        raise KeyError(f"Selected dimension '{dimension_col}' is not present in Set_Up_All_RF_data.")
    out['Dimension'] = out[dimension_col].astype(str)
    return out


def aggregate_overview(df, dimension_col, compare=('FY26','FY25')):
    df = add_dimension(df, dimension_col)
    value = df.groupby(['Dimension','fiscal_year'], as_index=False)['Publish.Dimension'].sum()
    value = value.pivot(index='Dimension', columns='fiscal_year', values='Publish.Dimension').reset_index()
    for fy in FY_ORDER:
        if fy not in value.columns:
            value[fy] = 0
    value = value[['Dimension'] + FY_ORDER].fillna(0)
    left, right = compare
    value['delta'] = value[left] - value[right]
    denom = value[right].replace(0, np.nan)
    value['delta.pc'] = (value['delta'] / denom).replace([np.inf, -np.inf], np.nan).fillna(0)
    base = left if left in value.columns else FY_ORDER[-1]
    total = value[base].sum()
    value[f'{base}.pc'] = 0 if total == 0 else value[base] / total

    spark = (df.groupby(['Dimension','period'], as_index=False)['Publish.Dimension'].sum()
               .sort_values(['Dimension','period'])
               .groupby('Dimension')['Publish.Dimension'].apply(list).rename('Quantity'))

    month = (df.groupby(['Dimension','calendar_month_abb','fiscal_year'], as_index=False)['Publish.Dimension'].sum())
    month = month.pivot(index=['Dimension','calendar_month_abb'], columns='fiscal_year', values='Publish.Dimension').reset_index()
    for fy in FY_ORDER:
        if fy not in month.columns:
            month[fy] = 0
    month['monthly.variation'] = month[left] - month[right]
    mv = (month.sort_values(['Dimension','calendar_month_abb'])
              .groupby('Dimension')['monthly.variation'].apply(list).rename('FY.Monthly.Variation.Quantity'))

    out = value.merge(spark, left_on='Dimension', right_index=True, how='left')\
               .merge(mv, left_on='Dimension', right_index=True, how='left')
    return out.sort_values(base, ascending=False)


def monthly_fy_df(df):
    out = df.groupby(['calendar_month_abb','fiscal_year'], as_index=False)['Publish.Dimension'].sum()
    out['calendar_month_abb'] = pd.Categorical(out['calendar_month_abb'], categories=MONTH_ORDER, ordered=True)
    out = out.sort_values('calendar_month_abb')
    wide = out.pivot(index='calendar_month_abb', columns='fiscal_year', values='Publish.Dimension').reset_index()
    for fy in FY_ORDER:
        if fy not in wide.columns:
            wide[fy] = 0
    return out, wide


def line_chart(df_long, years, title):
    p = df_long[df_long['fiscal_year'].isin(years)].copy()
    p['calendar_month_abb'] = pd.Categorical(p['calendar_month_abb'], categories=MONTH_ORDER, ordered=True)
    p = p.sort_values('calendar_month_abb')
    fig = px.line(p, x='calendar_month_abb', y='Publish.Dimension', color='fiscal_year', markers=True,
                  title=title, labels={'calendar_month_abb':'', 'Publish.Dimension':'in 9LC units'})
    fig.update_layout(height=340, legend_title=None, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def variation_bar(wide, left, right, title):
    p = wide.copy()
    p['variation'] = p[left] - p[right]
    fig = px.bar(p, x='calendar_month_abb', y='variation', title=title,
                 labels={'calendar_month_abb':'', 'variation':'in 9LC units'})
    colors = ['#22c55e' if v >= 0 else '#ef4444' for v in p['variation']]
    fig.update_traces(marker_color=colors, text=[fmt_int(v) for v in p['variation']], textposition='inside')
    fig.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def quarterly_chart(df_long):
    q = df_long.copy()
    q['quarter'] = q['calendar_month_abb'].map({m:'Q1' for m in MONTH_ORDER[:3]} | {m:'Q2' for m in MONTH_ORDER[3:6]} |
                                               {m:'Q3' for m in MONTH_ORDER[6:9]} | {m:'Q4' for m in MONTH_ORDER[9:12]})
    q = q.groupby(['quarter','fiscal_year'], as_index=False)['Publish.Dimension'].sum()
    q['quarter'] = pd.Categorical(q['quarter'], categories=['Q1','Q2','Q3','Q4'], ordered=True)
    q = q.sort_values('quarter')
    fig = px.bar(q, x='quarter', y='Publish.Dimension', color='fiscal_year', barmode='group',
                 title='Quarterly Sales by FY', labels={'Publish.Dimension':'in 9LC units'})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def sparkline(series, width=110, height=28):
    if not isinstance(series, (list, tuple, np.ndarray)):
        series = []
    if len(series) == 0:
        return None
    x = list(range(len(series)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=series, mode='lines', line=dict(width=1.5, color='#3b82f6')))
    fig.update_layout(height=height, width=width, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def sparkbar(series, width=110, height=28):
    if not isinstance(series, (list, tuple, np.ndarray)):
        series = []
    if len(series) == 0:
        return None
    colors = ['#60a5fa' if v >= 0 else '#f87171' for v in series]
    fig = go.Figure(go.Bar(x=list(range(len(series))), y=series, marker_color=colors))
    fig.update_layout(height=height, width=width, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def prepare_download(df, label):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    st.download_button(label, data=bio.getvalue(), file_name=f'{label.lower().replace(" ", "_")}.xlsx')


def styled_metric_table(df, compare_label='FY26 vs FY25', base_share='FY26.pc'):
    show = df.copy()
    rename_map = {
        'Dimension':'brand', 'FY24':'FY24 (9LC)', 'FY25':'FY25 (9LC)', 'FY26':'FY26 (9LC)', 'FY27':'FY27 (9LC)',
        base_share:'share of FY26 Volume (%)' if base_share=='FY26.pc' else 'share of FY25 Volume (%)',
        'delta':f'{compare_label} (9LC)', 'delta.pc':f'{compare_label} (%)'
    }
    show = show.rename(columns=rename_map)
    keep = [c for c in ['brand','FY24 (9LC)','FY25 (9LC)','FY26 (9LC)','FY27 (9LC)',
                        'share of FY26 Volume (%)','share of FY25 Volume (%)',
                        f'{compare_label} (9LC)', f'{compare_label} (%)'] if c in show.columns]
    fmt_df = show[keep].copy()
    for c in [col for col in fmt_df.columns if '(9LC)' in col]:
        fmt_df[c] = fmt_df[c].map(fmt_int)
    for c in [col for col in fmt_df.columns if '(%)' in col or 'Volume (%)' in col]:
        fmt_df[c] = fmt_df[c].map(fmt_pct)
    st.dataframe(fmt_df, use_container_width=True, hide_index=True)

    with st.expander('Mini charts'):
        for _, row in df.head(15).iterrows():
            c1, c2, c3, c4 = st.columns([2.8, 1.2, 1.2, 1.2])
            c1.write(str(row['Dimension']))
            with c2:
                fig = sparkline(row.get('Quantity', []))
                if fig: st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})
            with c3:
                fig = sparkbar(row.get('FY.Monthly.Variation.Quantity', []))
                if fig: st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})
            c4.write(fmt_int(row.get('delta', 0)))


def compare_publishes(df, filters, old_pub, new_pub, date_range, dimension_col):
    old_df = base_filtered_df(df, filters, old_pub, date_range=date_range)
    new_df = base_filtered_df(df, filters, new_pub, date_range=date_range)
    old_df = add_dimension(old_df, dimension_col)
    new_df = add_dimension(new_df, dimension_col)
    old_agg = old_df.groupby('Dimension', as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':'old_value'})
    new_agg = new_df.groupby('Dimension', as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':'new_value'})
    out = old_agg.merge(new_agg, on='Dimension', how='outer').fillna(0)
    out['delta'] = out['new_value'] - out['old_value']
    out['delta.pc'] = (out['delta'] / out['old_value'].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0)

    old_m = old_df.groupby(['calendar_month_abb'], as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':'old_value'})
    new_m = new_df.groupby(['calendar_month_abb'], as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':'new_value'})
    monthly = old_m.merge(new_m, on='calendar_month_abb', how='outer').fillna(0)
    monthly['delta'] = monthly['new_value'] - monthly['old_value']
    monthly['calendar_month_abb'] = pd.Categorical(monthly['calendar_month_abb'], categories=MONTH_ORDER, ordered=True)
    monthly = monthly.sort_values('calendar_month_abb')

    return out.sort_values('new_value', ascending=False), monthly


def show_comparison_charts(monthly, old_pub, new_pub):
    long = monthly.melt(id_vars='calendar_month_abb', value_vars=['old_value','new_value'], var_name='series', value_name='value')
    long['series'] = long['series'].map({'old_value':old_pub, 'new_value':new_pub})
    fig1 = px.line(long, x='calendar_month_abb', y='value', color='series', markers=True,
                   title='Monthly Projections Between Publishes', labels={'value':'in 9LC units', 'calendar_month_abb':''})
    st.plotly_chart(fig1, use_container_width=True)
    fig2 = px.bar(monthly, x='calendar_month_abb', y='delta', title='Upsides / Downsides vs Older Publish',
                  labels={'delta':'in 9LC units', 'calendar_month_abb':''})
    fig2.update_traces(marker_color=['#22c55e' if v >= 0 else '#ef4444' for v in monthly['delta']])
    st.plotly_chart(fig2, use_container_width=True)


def summarize_by_cols(df, group_cols, value_col='Publish.Dimension', sort_col=None):
    res = df.groupby(group_cols, as_index=False)[value_col].sum()
    if sort_col and sort_col in res.columns:
        res = res.sort_values(sort_col, ascending=False)
    return res


def render_table_and_download(df, name):
    st.dataframe(df, use_container_width=True, hide_index=True)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    st.download_button('Download', data=bio.getvalue(), file_name=f'{name}.xlsx', key=name)


def require_cols(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.warning(f"{label} is missing columns: {', '.join(missing)}")
        return False
    return True


st.title('Comparison between Sell IN RF Taiwan')

with st.sidebar:
    st.caption('Converted from the uploaded R Shiny app into Streamlit.')
    st.subheader('Data sources')
    set_up_file = st.file_uploader('Set_Up_All_RF_data.csv', type=['csv'])
    allocation_file = st.file_uploader('Allocation_data.csv', type=['csv'])
    pi_file = st.file_uploader('Set_Up_PI_data.csv', type=['csv'])

set_up_df = read_csv_any(set_up_file)
allocation_df = read_csv_any(allocation_file)
pi_df = read_csv_any(pi_file)

if set_up_df is None:
    st.info('Upload Set_Up_All_RF_data.csv to use the converted app.')
    st.stop()

set_up_df = normalize_text_cols(set_up_df, ['brand_quality','brand_quality_size','pig_description'])
set_up_df = ensure_month_order(set_up_df, 'calendar_month_abb')
if 'period' in set_up_df.columns:
    set_up_df['period'] = pd.to_datetime(set_up_df['period'], errors='coerce')
if allocation_df is not None and 'qty_alloc' in allocation_df.columns:
    allocation_df['qty_alloc'] = pd.to_numeric(allocation_df['qty_alloc'], errors='coerce').fillna(0)
if pi_df is not None and 'period' in pi_df.columns:
    pi_df['period'] = pd.to_datetime(pi_df['period'], errors='coerce')
    pi_df = pi_df[pi_df['period'] == pd.Timestamp('2025-06-01')]

# sidebar controls
with st.sidebar:
    st.subheader('Controls')
    publish_default = 'RF9_FY26' if 'RF9_FY26' in set_up_df.columns else next((c for c in PUBLISH_ORDER if c in set_up_df.columns), None)
    publish_choices = [c for c in PUBLISH_ORDER if c in set_up_df.columns]
    selected_publish = st.selectbox('Publish Month', publish_choices, index=publish_choices.index(publish_default) if publish_default in publish_choices else 0)
    period_range = st.slider('Fiscal Period', 1, 12, (1,12)) if 'fiscal_month' in set_up_df.columns else (1,12)
    old_publish_choices = publish_choices
    old_default = 'RF8_FY26' if 'RF8_FY26' in old_publish_choices else old_publish_choices[0]
    new_default = publish_default
    old_publish = st.selectbox('Oldest Publish Month', old_publish_choices, index=old_publish_choices.index(old_default), key='oldpub')
    new_publish = st.selectbox('Newest Publish Month', old_publish_choices, index=old_publish_choices.index(new_default), key='newpub')
    dmin = pd.Timestamp('2025-07-01')
    dmax = pd.Timestamp('2026-06-01')
    if 'period' in set_up_df.columns and set_up_df['period'].notna().any():
        dmin = set_up_df['period'].min().date()
        dmax = set_up_df['period'].max().date()
    date_range = st.date_input('Date range', (dmin, dmax)) if 'period' in set_up_df.columns else (dmin, dmax)
    threshold = st.date_input('Last 18 months threshold', pd.Timestamp('2024-07-01').date())

    st.subheader('Products')
    brand_sel = safe_multiselect('Brand', sorted(set_up_df['brand'].dropna().astype(str).unique()), key='brand') if 'brand' in set_up_df.columns else []
    bq_df = set_up_df[set_up_df['brand'].astype(str).isin(brand_sel)] if brand_sel and 'brand' in set_up_df.columns else set_up_df
    bq_sel = safe_multiselect('Brand Quality', sorted(bq_df['brand_quality'].dropna().astype(str).unique()), key='bq') if 'brand_quality' in bq_df.columns else []
    pd_df = bq_df[bq_df['brand_quality'].astype(str).isin(bq_sel)] if bq_sel and 'brand_quality' in bq_df.columns else bq_df
    pig_desc_sel = safe_multiselect('PIG Description', sorted(pd_df['pig_description'].dropna().astype(str).unique()), key='pigdesc') if 'pig_description' in pd_df.columns else []
    pc_df = set_up_df[set_up_df['pig_description'].astype(str).isin(pig_desc_sel)] if pig_desc_sel and 'pig_description' in set_up_df.columns else set_up_df
    pig_code_sel = safe_multiselect('PIG Code', sorted(pc_df['pig_code'].dropna().astype(str).unique()), key='pig') if 'pig_code' in pc_df.columns else []

    st.subheader('Channels')
    hcl_sel = safe_multiselect('High level Channel List', sorted(set_up_df['higher_channel_lst'].dropna().astype(str).unique()), key='hcl') if 'higher_channel_lst' in set_up_df.columns else []
    cg_df = set_up_df[set_up_df['higher_channel_lst'].astype(str).isin(hcl_sel)] if hcl_sel and 'higher_channel_lst' in set_up_df.columns else set_up_df
    cgn_sel = safe_multiselect('Customer Group Name', sorted(cg_df['customer_group_name'].dropna().astype(str).unique()), key='cgn') if 'customer_group_name' in cg_df.columns else []

filters = {
    'brand': brand_sel,
    'brand_quality': bq_sel,
    'pig_description': pig_desc_sel,
    'pig_code': pig_code_sel,
    'higher_channel_lst': hcl_sel,
    'customer_group_name': cgn_sel,
}

main_tabs = st.tabs(['Raw Data', 'Overview Market Demand', 'Comparison between Publishes', 'New | Live update',
                     'Check Forecasts on non active SKUs', 'Sell In per High Level Channels'])

with main_tabs[0]:
    sub = st.tabs(['Historical RF', 'Allocations', 'Opening Stocks FY26'])
    with sub[0]:
        hist = base_filtered_df(set_up_df, filters, selected_publish, period_range=period_range)
        st.dataframe(hist, use_container_width=True)
    with sub[1]:
        if allocation_df is not None:
            ad = allocation_df.copy()
            if pig_code_sel and 'pig_code' in ad.columns:
                ad = ad[ad['pig_code'].astype(str).isin(pig_code_sel)]
            st.dataframe(ad, use_container_width=True)
        else:
            st.info('Upload Allocation_data.csv to populate this tab.')
    with sub[2]:
        if pi_df is not None:
            od = pi_df.copy()
            if pig_code_sel and 'pig_code' in od.columns:
                od = od[od['pig_code'].astype(str).isin(pig_code_sel)]
            st.dataframe(od, use_container_width=True)
        else:
            st.info('Upload Set_Up_PI_data.csv to populate this tab.')

with main_tabs[1]:
    overview_df = base_filtered_df(set_up_df, filters, selected_publish, period_range=period_range)
    analysis_dim = st.radio('Select Dimension', options=[c for c in ['brand','brand_quality','brand_quality_size','pig_description','pig_code','higher_channel_lst','customer_groups_channel_lst','customer_group_name'] if c in overview_df.columns], horizontal=True)
    ov = aggregate_overview(overview_df, analysis_dim, compare=('FY26','FY25'))
    ov_nfy = aggregate_overview(overview_df, analysis_dim, compare=('FY27','FY26'))

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_block('FY26 vs FY25 (9LC)', fmt_int(ov['FY26'].sum() - ov['FY25'].sum()))
    with k2: kpi_block('FY26 vs FY25 (%)', fmt_pct((ov['FY26'].sum() - ov['FY25'].sum()) / ov['FY25'].sum() if ov['FY25'].sum() else 0))
    with k3: kpi_block('FY27 vs FY26 (9LC)', fmt_int(ov['FY27'].sum() - ov['FY26'].sum()))
    with k4: kpi_block('FY27 vs FY26 (%)', fmt_pct((ov['FY27'].sum() - ov['FY26'].sum()) / ov['FY26'].sum() if ov['FY26'].sum() else 0))

    c1, c2 = st.columns([4, 8])
    df_long, df_wide = monthly_fy_df(overview_df)
    with c1:
        subtabs = st.tabs(['FY26 & FY25', 'FY27 & FY26', 'FY24 : FY27', 'Month-Year', 'FY26 vs FY25 | FY27 vs FY26'])
        with subtabs[0]:
            st.plotly_chart(line_chart(df_long, ['FY25','FY26'], 'Monthly Sell IN : FY26 vs FY25'), use_container_width=True)
            st.plotly_chart(variation_bar(df_wide, 'FY26', 'FY25', 'Month Sell IN Variations : FY26 vs FY25'), use_container_width=True)
        with subtabs[1]:
            st.plotly_chart(line_chart(df_long, ['FY26','FY27'], 'Monthly Sell IN : FY27 vs FY26'), use_container_width=True)
            st.plotly_chart(variation_bar(df_wide, 'FY27', 'FY26', 'Month Sell IN Variations : FY27 vs FY26'), use_container_width=True)
        with subtabs[2]:
            st.plotly_chart(line_chart(df_long, FY_ORDER, 'Monthly Sell IN : FY24 to FY27'), use_container_width=True)
            st.plotly_chart(quarterly_chart(df_long), use_container_width=True)
        with subtabs[3]:
            fig = px.line(df_long.sort_values(['fiscal_year','calendar_month_abb']), x='period' if 'period' in overview_df.columns else 'calendar_month_abb',
                          y='Publish.Dimension', color='fiscal_year', title='Sell IN by Month-Year')
            st.plotly_chart(fig, use_container_width=True)
        with subtabs[4]:
            st.plotly_chart(line_chart(df_long, ['FY25','FY26'], 'Monthly Sell IN : FY26 vs FY25'), use_container_width=True)
            st.plotly_chart(line_chart(df_long, ['FY26','FY27'], 'Monthly Sell IN : FY27 vs FY26'), use_container_width=True)
    with c2:
        table_tabs = st.tabs(['FY26 vs FY25', 'FY27 vs FY26', 'FY26 vs FY25 | PIG code x Description'])
        with table_tabs[0]:
            st.markdown('FY26 vs FY25')
            st.caption('in 9L cases')
            styled_metric_table(ov, compare_label='FY26 vs FY25', base_share='FY26.pc')
            prepare_download(ov, 'react_current_RF_data')
        with table_tabs[1]:
            st.markdown('FY27 vs FY26')
            st.caption('in 9L cases')
            styled_metric_table(ov_nfy, compare_label='FY27 vs FY26', base_share='FY26.pc')
            prepare_download(ov_nfy, 'react_current_RF_NFY_vs_CFY_data')
        with table_tabs[2]:
            pig_df = aggregate_overview(overview_df, 'pig_code', compare=('FY26','FY25')) if 'pig_code' in overview_df.columns else pd.DataFrame()
            if not pig_df.empty and 'pig_description' in overview_df.columns:
                pig_desc = overview_df[['pig_code','pig_description']].dropna().drop_duplicates().astype(str)
                pig_df = pig_df.merge(pig_desc, left_on='Dimension', right_on='pig_code', how='left')
                pig_df = pig_df[['Dimension','pig_description','FY24','FY25','FY26','FY27','FY26.pc','delta','delta.pc','Quantity','FY.Monthly.Variation.Quantity']]
                pig_df = pig_df.rename(columns={'Dimension':'pig_code'})
                show = pig_df[['pig_code','pig_description','FY24','FY25','FY26','FY27','FY26.pc','delta','delta.pc']].copy()
                for col in ['FY24','FY25','FY26','FY27','delta']:
                    show[col] = show[col].map(fmt_int)
                for col in ['FY26.pc','delta.pc']:
                    show[col] = show[col].map(fmt_pct)
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.info('Required columns for PIG code x Description are not available.')

with main_tabs[2]:
    compare_df = base_filtered_df(set_up_df, filters, selected_publish, date_range=date_range)
    comparison_dim = st.radio('Select Comparison Dimension', options=[c for c in ['brand','brand_quality','brand_quality_size','pig_description','pig_code','higher_channel_lst','customer_groups_channel_lst','customer_group_name'] if c in compare_df.columns], horizontal=True)
    diff, monthly = compare_publishes(set_up_df, filters, old_publish, new_publish, date_range, comparison_dim)
    c1, c2 = st.columns([5,7])
    with c1:
        delta_total = monthly['delta'].sum()
        pct_total = delta_total / monthly['old_value'].sum() if monthly['old_value'].sum() else 0
        kpi_block(f'{new_publish} vs {old_publish} (9LC)', fmt_int(delta_total))
        kpi_block(f'{new_publish} vs {old_publish} (%)', fmt_pct(pct_total))
        show_comparison_charts(monthly, old_publish, new_publish)
    with c2:
        tabs2 = st.tabs(['by Dimension', 'PIG code x Customer Group', 'PIG code x Description'])
        with tabs2[0]:
            st.caption('Variations between Publishes in 9L cases')
            show = diff.rename(columns={'Dimension':comparison_dim, 'old_value':old_publish, 'new_value':new_publish, 'delta':f'{new_publish} - {old_publish}', 'delta.pc':'delta %'})
            render_table_and_download(show, 'react_Demand_Variations_data')
        with tabs2[1]:
            if require_cols(set_up_df, ['pig_code','customer_group_name'], 'Set_Up_All_RF_data'):
                old_df = base_filtered_df(set_up_df, filters, old_publish, date_range=date_range)
                new_df = base_filtered_df(set_up_df, filters, new_publish, date_range=date_range)
                old_g = old_df.groupby(['pig_code','customer_group_name'], as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':old_publish})
                new_g = new_df.groupby(['pig_code','customer_group_name'], as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':new_publish})
                pg = old_g.merge(new_g, on=['pig_code','customer_group_name'], how='outer').fillna(0)
                pg['delta'] = pg[new_publish] - pg[old_publish]
                pg['delta.pc'] = (pg['delta'] / pg[old_publish].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0)
                render_table_and_download(pg, 'react_Demand_Variations_PIG_x_CustomerGroup_data')
        with tabs2[2]:
            if require_cols(set_up_df, ['pig_code','pig_description'], 'Set_Up_All_RF_data'):
                old_df = base_filtered_df(set_up_df, filters, old_publish, date_range=date_range)
                new_df = base_filtered_df(set_up_df, filters, new_publish, date_range=date_range)
                old_g = old_df.groupby(['pig_code','pig_description'], as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':old_publish})
                new_g = new_df.groupby(['pig_code','pig_description'], as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':new_publish})
                pg = old_g.merge(new_g, on=['pig_code','pig_description'], how='outer').fillna(0)
                pg['delta'] = pg[new_publish] - pg[old_publish]
                pg['delta.pc'] = (pg['delta'] / pg[old_publish].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0)
                render_table_and_download(pg, 'react_Demand_Variations_PIG_x_Description_data')

with main_tabs[3]:
    st.markdown('Regular check of items to be updated')
    st.caption('moving from Status New to Status Live if actual sales happened more than 18 months ago')
    live_df = base_filtered_df(set_up_df, filters, selected_publish)
    if require_cols(live_df, ['sales_start_horizon_tag'], 'Set_Up_All_RF_data'):
        threshold_ts = pd.to_datetime(threshold)
        if 'period' in live_df.columns:
            grp = live_df.groupby(['pig_code','pig_description','sales_start_horizon_tag'], as_index=False)['period'].max()
            grp['suggested_status'] = np.where(grp['period'] <= threshold_ts, 'Live', grp['sales_start_horizon_tag'])
            grp = grp[grp['sales_start_horizon_tag'].astype(str).str.lower().eq('new') & grp['suggested_status'].eq('Live')]
            render_table_and_download(grp, 'react_status_change_data')
        else:
            st.info('Column period is required for this tab.')

with main_tabs[4]:
    st.markdown('Overview Total Sell IN Forecasts')
    st.caption('by Central Status')
    chk = base_filtered_df(set_up_df, filters, selected_publish, date_range=date_range)
    status_col = next((c for c in ['stock_tag','ending_FY_tag','sales_start_horizon_tag','allocation_tag'] if c in chk.columns), None)
    if status_col:
        overview = chk.groupby(status_col, as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':'sell_in'})
        overview['sell_in'] = overview['sell_in'].map(fmt_int)
        st.dataframe(overview, use_container_width=True, hide_index=True)
        inactive = chk[chk[status_col].astype(str).str.lower().isin(['dead','delisted'])] if status_col else chk.iloc[0:0]
        if not inactive.empty:
            detail = inactive.groupby(['pig_code','pig_description','calendar_month_abb'], as_index=False)['Publish.Dimension'].sum()
            st.markdown('Details monthly Sell IN Forecasts by PIG code')
            render_table_and_download(detail, 'react_check_forecasts_active_skus_data')
        else:
            st.info('No dead or delisted SKUs with forecast found under the current filters.')
    else:
        st.info('No central status field found in the dataset.')

with main_tabs[5]:
    hl = base_filtered_df(set_up_df, filters, selected_publish, date_range=date_range)
    if require_cols(hl, ['higher_channel_lst'], 'Set_Up_All_RF_data'):
        total = hl.groupby('higher_channel_lst', as_index=False)['Publish.Dimension'].sum().rename(columns={'Publish.Dimension':'Total Sell IN'})
        st.markdown('Total Sell IN Forecasts per High Level Channels')
        total['Total Sell IN'] = total['Total Sell IN'].map(fmt_int)
        st.dataframe(total, use_container_width=True, hide_index=True)

        monthly = hl.groupby(['higher_channel_lst','calendar_month_abb'], as_index=False)['Publish.Dimension'].sum()
        wide = monthly.pivot(index='higher_channel_lst', columns='calendar_month_abb', values='Publish.Dimension').reset_index()
        for m in MONTH_ORDER:
            if m not in wide.columns:
                wide[m] = 0
        wide = wide[['higher_channel_lst'] + MONTH_ORDER]
        for m in MONTH_ORDER:
            wide[m] = wide[m].map(fmt_int)
        st.markdown('Monthly Sell IN Forecasts')
        render_table_and_download(wide, 'react_high_level_sell_in_data')

        by_month = hl.groupby('calendar_month_abb', as_index=False)['Publish.Dimension'].sum().sort_values('calendar_month_abb')
        by_month['Publish.Dimension'] = by_month['Publish.Dimension'].map(fmt_int)
        st.markdown('Monthly Sell IN Forecasts | to fill up the CTS dashboard')
        render_table_and_download(by_month.rename(columns={'Publish.Dimension':'Sell IN'}), 'react_month_abb_sell_in_data')
