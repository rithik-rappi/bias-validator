import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="Forecast Bias Dashboard", page_icon="📦")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
    .metric-box-red {
        background: #fff5f5; border: 1px solid #fc8181;
        border-radius: 8px; padding: 12px 16px;
    }
    .metric-box-orange {
        background: #fffaf0; border: 1px solid #f6ad55;
        border-radius: 8px; padding: 12px 16px;
    }
    .kpi-label  { font-size: 11px; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
    .kpi-value  { font-size: 22px; font-weight: 700; color: #212529; }
    .kpi-value-red    { font-size: 22px; font-weight: 700; color: #e53e3e; }
    .kpi-value-orange { font-size: 22px; font-weight: 700; color: #dd6b20; }
    .section-header {
        font-size: 13px; font-weight: 700; color: #495057;
        text-transform: uppercase; letter-spacing: .6px;
        border-bottom: 2px solid #dee2e6; padding-bottom: 4px; margin-bottom: 10px;
    }
    div[data-testid="stDataFrame"] table { font-size: 12px; }
    .stSelectbox label, .stMultiSelect label { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(file):
    df = pd.read_parquet(file)
    for _dc in ['plan_start_date', 'plan_end_date']:
        if _dc in df.columns:
            df[_dc] = pd.to_datetime(df[_dc], errors='coerce')
    for col in ['quantity_sold','order_quantity','forecast','forecast_ss','error_capped','bias',
                'swa','doi','error_ss_capped','bias_ss','availability',
                'bias%','bias_ss%','z_sl','Z_SL','rmse_component','forecast_error_component',
                'lead_time_variance_component','bias_adjustment','forecast_error_percent_bias',
                'uncapped_safety_stock','new_ss_raw','new_ss_capped','original_safety_stock',
                'average_sales_14day']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # derive excess
    if 'forecast_ss' in df.columns and 'quantity_sold' in df.columns:
        df['excess'] = df['forecast_ss'] - df['quantity_sold']
    # derive step_diff (SS coverage bucket shift) if not already present
    if 'step_diff' not in df.columns and 'coverage_bucket' in df.columns and 'coverage_bucket_ss' in df.columns:
        _ordering = {
            '<-300%': 1, '-300% to -160%': 2, '-160% to -70%': 3, '-70% to -30%': 4,
            '-30% to 0%': 5, '0% to 30%': 6, '30% to 70%': 7, '70% to 160%': 8,
            '160% to 300%': 9, '>300%': 10
        }
        df['_cb_val']    = df['coverage_bucket'].map(_ordering)
        df['_cb_ss_val'] = df['coverage_bucket_ss'].map(_ordering)
        df['step_diff']  = df['_cb_ss_val'] - df['_cb_val']
        df.drop(columns=['_cb_val', '_cb_ss_val'], inplace=True)
    return df

uploaded = st.file_uploader("Upload plan_data parquet file", type=["parquet"])
if uploaded is None:
    st.info("⬆️  Upload the parquet file to begin.")
    st.stop()

plan_data = load_data(uploaded)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS  (applied to plan_data → filtered; used by both tabs)
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("🔧 Global Filters")

filter_qty = st.sidebar.checkbox("Filter: quantity_sold > 0", value=False)
if filter_qty:
    plan_data = plan_data[plan_data['quantity_sold'] > 0]

filter_avb = st.sidebar.checkbox("Filter: availability ≥ 85%", value=False)
if filter_avb and 'availability' in plan_data.columns:
    plan_data = plan_data[plan_data['availability'] >= 85]

# SS coverage shift filter
if 'step_diff' in plan_data.columns:
    _sd_min = int(plan_data['step_diff'].dropna().min()) if plan_data['step_diff'].notna().any() else 0
    _sd_max = int(plan_data['step_diff'].dropna().max()) if plan_data['step_diff'].notna().any() else 10
    filter_step = st.sidebar.number_input(
        "SS Coverage Shift ≥ (buckets)",
        min_value=_sd_min, max_value=_sd_max,
        value=0, step=1,
        help="Keep only plans where SS pushed the coverage bucket up by at least this many steps. "
             "E.g. '2' means the SS bumped coverage at least 2 bucket levels higher than the base forecast."
    )
    if filter_step > 0:
        plan_data = plan_data[plan_data['step_diff'] >= filter_step]

# inflatable filter
_infl_col = next((c for c in ['infaltable','Infaltable','inflatable','Inflatable','is_inflatable']
                  if c in plan_data.columns), None)
if _infl_col:
    filter_infl = st.sidebar.radio(
        "Inflatable filter", ["All", "Inflatable only", "Non-Inflatable only"],
        index=0, key="infl_filter"
    )
    # handle bool, int (1/0), or string ('True'/'False') column values
    _true_vals  = [True, 1, 'True',  'true',  'TRUE',  '1', 'yes', 'Yes']
    _false_vals = [False, 0, 'False', 'false', 'FALSE', '0', 'no',  'No']
    if filter_infl == "Inflatable only":
        plan_data = plan_data[plan_data[_infl_col].isin(_true_vals)]
    elif filter_infl == "Non-Inflatable only":
        plan_data = plan_data[plan_data[_infl_col].isin(_false_vals)]
else:
    st.sidebar.caption("ℹ️ No inflatable column found")

FILTER_COLS = ['sku_classification','time_series_class_day','category_name',
               'city_name','abc_class','xyz_class','coverage_bucket','coverage_bucket_ss']

active_filters = {}
for col in FILTER_COLS:
    if col in plan_data.columns:
        opts = sorted(plan_data[col].dropna().unique().tolist())
        sel = st.sidebar.multiselect(col, opts, default=[], key=f"filter_{col}")
        if sel:
            active_filters[col] = sel

filtered = plan_data.copy()
for col, vals in active_filters.items():
    filtered = filtered[filtered[col].isin(vals)]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Rows after filters:** `{len(filtered):,}`")

DIM_COLS = [c for c in FILTER_COLS if c in plan_data.columns]
# add inflatable column to group-by options if present
if _infl_col and _infl_col not in DIM_COLS:
    DIM_COLS = DIM_COLS + [_infl_col]
groupby_dim = st.sidebar.selectbox("📐 Group-by dimension", DIM_COLS, index=0)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def raw_data_expander(df, label="raw_data", key_suffix=""):
    """Adds a collapsible raw data viewer with CSV download under any section."""
    with st.expander("📋 View raw data", expanded=False):
        st.dataframe(df, use_container_width=True, height=350)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv,
            file_name=f"{label}.csv",
            mime="text/csv",
            key=f"dl_{label}_{key_suffix}"
        )

def compute_metrics(df, total_qs_for_excess=None):
    qs  = df['quantity_sold'].sum()
    fc  = df['forecast'].sum()
    fss = df['forecast_ss'].sum()
    oq  = df['order_quantity'].sum() if 'order_quantity' in df.columns else np.nan
    wmape_fc  = 100 * df['error_capped'].sum()    / qs if qs else np.nan
    bias_fc   = 100 * df['bias'].sum()             / qs if qs else np.nan
    swa_fc    = df['swa'].mean()                        if 'swa' in df.columns else np.nan
    doi_fc    = df['doi'].mean()                        if 'doi' in df.columns else np.nan
    wmape_ss  = 100 * df['error_ss_capped'].sum()  / qs if qs else np.nan
    bias_ss   = 100 * df['bias_ss'].sum()          / qs if qs else np.nan
    excess    = df['excess'].sum()                      if 'excess' in df.columns else np.nan
    denom     = total_qs_for_excess if total_qs_for_excess else qs
    excess_impact = 100 * excess / denom                if (not np.isnan(excess) and denom) else np.nan
    return dict(qty_sold=qs, fc=fc, fc_ss=fss, oq=oq,
                wmape_fc=wmape_fc, bias_fc=bias_fc, swa_fc=swa_fc, doi_fc=doi_fc,
                wmape_ss=wmape_ss, bias_ss=bias_ss,
                excess=excess, excess_impact=excess_impact)

def make_metrics_table(df):
    total_qs = df['quantity_sold'].sum()
    avb_high = df[df['availability'] >= 85] if 'availability' in df.columns else df
    avb_low  = df[df['availability'] <  85] if 'availability' in df.columns else df.iloc[0:0]
    rows = []
    for label, subset in [("All", df), ("Avb ≥ 85%", avb_high), ("Avb < 85%", avb_low)]:
        m = compute_metrics(subset, total_qs_for_excess=total_qs)
        rows.append({
            "Segment":          label,
            "Filters":          subset['plan_index'].nunique() if 'plan_index' in subset.columns else len(subset),
            "Qty Sold":         int(m['qty_sold']),
            "Order Qty":        int(m['oq'])              if not np.isnan(m['oq'])       else "—",
            "FC":               int(m['fc']),
            "FC+SS":            int(m['fc_ss']),
            "Excess":           int(m['excess'])          if not np.isnan(m['excess'])   else "—",
            "Excess Impact%":   m['excess_impact'],
            "wMAPE [FC]":       f"{m['wmape_fc']:.2f}%"  if not np.isnan(m['wmape_fc']) else "—",
            "Bias [FC]":        m['bias_fc'],
            "SWA":              f"{m['swa_fc']:.2f}%"    if not np.isnan(m['swa_fc'])   else "—",
            "DOI":              f"{m['doi_fc']:.2f}"     if not np.isnan(m['doi_fc'])    else "—",
            "wMAPE [FC+SS]":    f"{m['wmape_ss']:.2f}%"  if not np.isnan(m['wmape_ss']) else "—",
            "Bias [FC+SS]":     m['bias_ss'],
        })
    return pd.DataFrame(rows)

def style_bias(val):
    if isinstance(val, float):
        color = "#e53e3e" if val > 0 else "#2b6cb0"
        return f"color: {color}; font-weight: 600;"
    return ""

def color_bias_col(s):
    return [style_bias(v) for v in s]

def of_analysis(df, dim):
    needed = ['plan_index','storereferenceid','warehouseid', dim, 'bias%','bias_ss%','bias','bias_ss']
    plan_df = df[[c for c in needed if c in df.columns]].copy()
    if 'storereferenceid' not in plan_df.columns or 'warehouseid' not in plan_df.columns:
        return pd.DataFrame(), plan_df, pd.DataFrame()

    plan_count = plan_df.groupby(['storereferenceid','warehouseid']).size().reset_index(name='plan_count')
    plan_df = plan_df.merge(plan_count, on=['storereferenceid','warehouseid'], how='left')

    pstv = (plan_df[plan_df['bias%'] > 0]
            .groupby(['storereferenceid','warehouseid']).size()
            .reset_index(name='pstv_FC_bias_count'))
    plan_df = plan_df.merge(pstv, on=['storereferenceid','warehouseid'], how='left')
    plan_df['pstv_FC_bias_count'] = plan_df['pstv_FC_bias_count'].fillna(0)
    plan_df['pstv_plans_ratio']   = plan_df['pstv_FC_bias_count'] / plan_df['plan_count']
    consistent_OF = plan_df[plan_df['pstv_plans_ratio'] > 0.8]

    OF_filters = (
        plan_df.groupby(dim).size().reset_index(name='plan-filter_combinations')
        .merge(consistent_OF.groupby(dim).size().reset_index(name='OF_plan-filter_combinations'), on=dim, how='left')
    )
    OF_filters['OF_plan-filter_combinations'] = OF_filters['OF_plan-filter_combinations'].fillna(0).astype(int)
    OF_filters['filter%'] = (OF_filters['OF_plan-filter_combinations'] / OF_filters['plan-filter_combinations'] * 100).round(1)

    OF_bias = (
        plan_df.groupby(dim)['bias'].sum().reset_index(name='plan-filter_bias')
        .merge(consistent_OF.groupby(dim)['bias'].sum().reset_index(name='OF_plan-filter_bias'), on=dim, how='left')
    )
    OF_bias['OF_plan-filter_bias'] = OF_bias['OF_plan-filter_bias'].fillna(0).round(0).astype(int)
    OF_bias['plan-filter_bias']    = OF_bias['plan-filter_bias'].round(0).astype(int)
    OF_bias['bias_contri%'] = (OF_bias['OF_plan-filter_bias'] / OF_bias['plan-filter_bias'].replace(0, np.nan) * 100).round(1)

    return OF_filters.merge(OF_bias, on=dim, how='left'), plan_df, consistent_OF

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📦 Forecast Bias & Safety Stock Dashboard")
tab_main, tab_ss = st.tabs(["📊 Forecast & Bias", "🧮 Safety Stock Decomposition"])

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — FORECAST & BIAS                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_main:

    # ── KPI bar ───────────────────────────────────────────────────────────────
    qs_total    = filtered['quantity_sold'].sum()
    bias_agg    = 100 * filtered['bias'].sum()    / qs_total if qs_total else 0
    bias_ss_agg = 100 * filtered['bias_ss'].sum() / qs_total if qs_total else 0
    excess_total = filtered['excess'].sum() if 'excess' in filtered.columns else 0
    excess_impact_total = 100 * excess_total / qs_total if qs_total else 0

    col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)
    with col_k1:
        st.markdown(f"""<div class="metric-box"><div class="kpi-label">Total Qty Sold</div>
            <div class="kpi-value">{int(qs_total):,}</div></div>""", unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""<div class="metric-box"><div class="kpi-label">Total Forecast</div>
            <div class="kpi-value">{int(filtered['forecast'].sum()):,}</div></div>""", unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""<div class="metric-box"><div class="kpi-label">Total FC+SS</div>
            <div class="kpi-value">{int(filtered['forecast_ss'].sum()):,}</div></div>""", unsafe_allow_html=True)
    with col_k4:
        st.markdown(f"""<div class="metric-box-orange"><div class="kpi-label">Total Excess</div>
            <div class="kpi-value-orange">{int(excess_total):,} ({excess_impact_total:+.1f}%)</div></div>""", unsafe_allow_html=True)
    with col_k5:
        cc = "kpi-value-red" if bias_agg > 0 else "kpi-value"
        st.markdown(f"""<div class="metric-box-red"><div class="kpi-label">Bias% [FC]</div>
            <div class="{cc}">{bias_agg:+.2f}%</div></div>""", unsafe_allow_html=True)
    with col_k6:
        cc2 = "kpi-value-red" if bias_ss_agg > 0 else "kpi-value"
        st.markdown(f"""<div class="metric-box-red"><div class="kpi-label">Bias% [FC+SS]</div>
            <div class="{cc2}">{bias_ss_agg:+.2f}%</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 1: Metrics table ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Accuracy Metrics by Availability</div>', unsafe_allow_html=True)
    metrics_df = make_metrics_table(filtered)
    styled = (metrics_df.style
        .format({"Qty Sold":"{:,}","Order Qty":"{:,}","FC":"{:,}","FC+SS":"{:,}","Excess":"{:,}",
                 "Excess Impact%":"{:+.2f}%","Bias [FC]":"{:+.2f}%","Bias [FC+SS]":"{:+.2f}%"})
        .apply(color_bias_col, subset=["Bias [FC]","Bias [FC+SS]"])
        .hide(axis='index')
    )
    st.dataframe(styled, use_container_width=True, height=160)
    raw_data_expander(filtered, "metrics_raw", "metrics")

    st.markdown("---")

    # ── Section 2: OF Analysis + Quadrant ────────────────────────────────────
    st.markdown(f'<div class="section-header">🔍 Over-Forecast Analysis by {groupby_dim}</div>', unsafe_allow_html=True)

    try:
        of_result, plan_df_full, consistent_OF_df = of_analysis(filtered, groupby_dim)
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown(f"**Consistent OF (>80% positive bias plans) — grouped by `{groupby_dim}`**")

            def style_of_table(df):
                fmt = {'filter%':'{:.1f}%','bias_contri%':'{:.1f}%',
                       'plan-filter_bias':'{:,}','OF_plan-filter_bias':'{:,}'}
                s = df.style.format(fmt).hide(axis='index')
                if 'filter%'      in df.columns: s = s.background_gradient(cmap='RdYlGn_r', subset=['filter%'],      vmin=0, vmax=100)
                if 'bias_contri%' in df.columns: s = s.background_gradient(cmap='RdYlGn_r', subset=['bias_contri%'], vmin=0, vmax=100)
                return s

            if not of_result.empty:
                st.dataframe(style_of_table(of_result), use_container_width=True, height=300)
            else:
                st.warning("OF analysis requires storereferenceid / warehouseid columns.")

            dim_values = sorted(filtered[groupby_dim].dropna().unique().tolist()) if groupby_dim in filtered.columns else []
            selected_dim_val = st.selectbox(
                f"🔎 Select {groupby_dim} value to drill down",
                ["(All)"] + [str(v) for v in dim_values], key="dim_drill"
            )

        with col_right:
            st.markdown("**Bias% vs Bias_SS% — 4-Quadrant Scatter**")
            scatter_df = filtered.copy()
            if 'bias%' in scatter_df.columns and 'bias_ss%' in scatter_df.columns:
                scatter_df = scatter_df.dropna(subset=['bias%','bias_ss%'])
                scatter_plot_df = (scatter_df[scatter_df[groupby_dim].astype(str) == selected_dim_val]
                                   if selected_dim_val != "(All)" else scatter_df)
                if len(scatter_plot_df) > 5000:
                    scatter_plot_df = scatter_plot_df.sample(5000, random_state=42)

                if len(scatter_plot_df) == 0:
                    st.warning("No data to plot.")
                else:
                    xq = scatter_plot_df['bias%'].quantile([0.01,0.99]).values
                    yq = scatter_plot_df['bias_ss%'].quantile([0.01,0.99]).values
                    x_lo, x_hi = min(xq[0],0)*1.1-1, max(xq[1],0)*1.1+1
                    y_lo, y_hi = min(yq[0],0)*1.1-1, max(yq[1],0)*1.1+1

                    fig_q = go.Figure()
                    for x0,x1,y0,y1,color,lbl in [
                        (0,x_hi,0,y_hi,'rgba(252,129,129,0.10)','OF / OF+SS'),
                        (x_lo,0,0,y_hi,'rgba(107,194,107,0.10)','UF / OF+SS'),
                        (0,x_hi,y_lo,0,'rgba(255,166,77,0.10)', 'OF / UF+SS'),
                        (x_lo,0,y_lo,0,'rgba(100,149,237,0.10)','UF / UF+SS'),
                    ]:
                        fig_q.add_shape(type="rect",x0=x0,x1=x1,y0=y0,y1=y1,fillcolor=color,line_width=0,layer="below")
                        fig_q.add_annotation(x=(x0+x1)/2,y=(y0+y1)/2,text=lbl,showarrow=False,font=dict(size=9,color='#888'))
                    fig_q.add_hline(y=0,line_dash="dash",line_color="gray",line_width=1)
                    fig_q.add_vline(x=0,line_dash="dash",line_color="gray",line_width=1)

                    if groupby_dim in scatter_plot_df.columns and selected_dim_val == "(All)":
                        for cv in scatter_plot_df[groupby_dim].unique():
                            sub = scatter_plot_df[scatter_plot_df[groupby_dim]==cv]
                            fig_q.add_trace(go.Scatter(x=sub['bias%'],y=sub['bias_ss%'],mode='markers',
                                name=str(cv),marker=dict(size=4,opacity=0.55)))
                    else:
                        fig_q.add_trace(go.Scatter(x=scatter_plot_df['bias%'],y=scatter_plot_df['bias_ss%'],
                            mode='markers',name=str(selected_dim_val),marker=dict(size=4,opacity=0.55,color='steelblue')))

                    fig_q.update_layout(height=370,margin=dict(l=40,r=20,t=20,b=40),plot_bgcolor='white',
                        xaxis=dict(title="Bias% [FC]",showgrid=True,gridcolor='#f0f0f0',range=[x_lo,x_hi]),
                        yaxis=dict(title="Bias% [FC+SS]",showgrid=True,gridcolor='#f0f0f0',range=[y_lo,y_hi]),
                        legend=dict(font=dict(size=10),itemsizing='constant'))
                    st.plotly_chart(fig_q, use_container_width=True)
            else:
                st.warning("Columns `bias%` / `bias_ss%` not found.")

        # ── Coverage Matrix — always shown (whole set or drilldown) ──────────
        st.markdown("---")
        if selected_dim_val == "(All)":
            cov_df      = filtered
            cov_label   = "All (filtered)"
        else:
            cov_df      = filtered[filtered[groupby_dim].astype(str) == selected_dim_val]
            cov_label   = f"{groupby_dim} = {selected_dim_val}"

        if 'coverage_bucket' in filtered.columns and 'coverage_bucket_ss' in filtered.columns:
            st.markdown(f'<div class="section-header">🗺️ Coverage Bucket Matrix — {cov_label}</div>', unsafe_allow_html=True)
            matrix = pd.crosstab(cov_df['coverage_bucket'], cov_df['coverage_bucket_ss'])
            st.dataframe(matrix.style.background_gradient(cmap='YlOrRd').format("{:,}"), use_container_width=True)

        # ── Drilldown KPIs + metrics — only when a value is selected ─────────
        if selected_dim_val != "(All)":
            drill_df = cov_df  # already filtered above
            st.markdown(f'<div class="section-header">🔬 Drill-down: {groupby_dim} = "{selected_dim_val}"</div>', unsafe_allow_html=True)
            qs_d  = drill_df['quantity_sold'].sum()
            b_d   = 100 * drill_df['bias'].sum()    / qs_d if qs_d else 0
            bs_d  = 100 * drill_df['bias_ss'].sum() / qs_d if qs_d else 0
            exc_d = drill_df['excess'].sum()                if 'excess' in drill_df.columns else 0
            exc_impact_d = 100 * exc_d / qs_d               if qs_d else 0

            _, kd1, kd2, kd3, kd4 = st.columns([2,1,1,1,1])
            with kd1:
                st.markdown(f"""<div class="metric-box"><div class="kpi-label">Qty Sold</div>
                    <div class="kpi-value">{int(qs_d):,}</div></div>""", unsafe_allow_html=True)
            with kd2:
                cc = "kpi-value-red" if b_d > 0 else "kpi-value"
                st.markdown(f"""<div class="metric-box-red"><div class="kpi-label">Bias% [FC]</div>
                    <div class="{cc}">{b_d:+.2f}%</div></div>""", unsafe_allow_html=True)
            with kd3:
                cc2 = "kpi-value-red" if bs_d > 0 else "kpi-value"
                st.markdown(f"""<div class="metric-box-red"><div class="kpi-label">Bias% [FC+SS]</div>
                    <div class="{cc2}">{bs_d:+.2f}%</div></div>""", unsafe_allow_html=True)
            with kd4:
                st.markdown(f"""<div class="metric-box-orange"><div class="kpi-label">Excess Impact%</div>
                    <div class="kpi-value-orange">{exc_impact_d:+.2f}%</div></div>""", unsafe_allow_html=True)

            st.markdown(f"**Accuracy metrics for {groupby_dim} = {selected_dim_val}**")
            sub_metrics = make_metrics_table(drill_df)
            st.dataframe(
                sub_metrics.style
                    .format({"Qty Sold":"{:,}","FC":"{:,}","FC+SS":"{:,}","Excess":"{:,}",
                             "Excess Impact%":"{:+.2f}%","Bias [FC]":"{:+.2f}%","Bias [FC+SS]":"{:+.2f}%"})
                    .apply(color_bias_col, subset=["Bias [FC]","Bias [FC+SS]"])
                    .hide(axis='index'),
                use_container_width=True, height=160
            )
    except Exception as e:
        st.error(f"Error: {e}")

    st.markdown("---")

    # ── Section 3: DOW Pattern ────────────────────────────────────────────────
    _has_date_src = any(c in filtered.columns for c in ['plan_end_date','plan_start_date'])
    if _has_date_src:
        st.markdown('<div class="section-header">📅 Weekend / Weekday Safety Stock Pattern</div>', unsafe_allow_html=True)

        wd_df = filtered.copy()
        wd_df['plan_start_date'] = pd.to_datetime(wd_df['plan_start_date'], errors='coerce')
        wd_df['plan_end_date']   = pd.to_datetime(wd_df['plan_end_date'],   errors='coerce')
        wd_df['date'] = wd_df.apply(
            lambda r: pd.date_range(r['plan_start_date'], r['plan_end_date'], freq='D')
                      if pd.notna(r['plan_start_date']) and pd.notna(r['plan_end_date']) else [],
            axis=1
        )
        wd_df = wd_df.explode('date').reset_index(drop=True)
        wd_df['duration_days'] = (wd_df['plan_end_date'] - wd_df['plan_start_date']).dt.days + 1
        for _col in ['quantity_sold','forecast','forecast_ss','bias','bias_ss','excess','error_capped','error_ss_capped']:
            if _col in wd_df.columns:
                wd_df[_col] = wd_df[_col] / wd_df['duration_days']
        wd_df['weekday_name'] = wd_df['date'].dt.day_name()

        DOW_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        wk = (wd_df.groupby('weekday_name')
              .agg(qty_sold=('quantity_sold','sum'), forecast=('forecast','sum'),
                   forecast_ss=('forecast_ss','sum'), bias_sum=('bias','sum'),
                   bias_ss_sum=('bias_ss','sum'))
              .reset_index())
        wk['bias_pct']    = 100 * wk['bias_sum']    / wk['qty_sold'].replace(0, np.nan)
        wk['bias_ss_pct'] = 100 * wk['bias_ss_sum'] / wk['qty_sold'].replace(0, np.nan)
        wk['ss_delta']    = wk['forecast_ss'] - wk['forecast']
        wk = wk[wk['weekday_name'].isin(DOW_ORDER)].copy()
        wk['weekday_name'] = pd.Categorical(wk['weekday_name'], categories=DOW_ORDER, ordered=True)
        wk = wk.sort_values('weekday_name')

        wk_col1, wk_col2 = st.columns(2)
        with wk_col1:
            fig_wk = go.Figure()
            fig_wk.add_trace(go.Bar(x=wk['weekday_name'].astype(str), y=wk['bias_pct'],    name='Bias% [FC]',    marker_color='steelblue', opacity=0.8))
            fig_wk.add_trace(go.Bar(x=wk['weekday_name'].astype(str), y=wk['bias_ss_pct'], name='Bias% [FC+SS]', marker_color='tomato',    opacity=0.8))
            fig_wk.update_layout(title="Bias% by Day of Week", barmode='group', height=300,
                margin=dict(l=40,r=20,t=40,b=40), yaxis_title="Bias%", plot_bgcolor='white',
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=True, zerolinecolor='#aaa'))
            st.plotly_chart(fig_wk, use_container_width=True)
        with wk_col2:
            bar_colors = ['rgba(255,140,60,0.85)' if d in ['Saturday','Sunday'] else 'rgba(80,160,200,0.75)'
                          for d in wk['weekday_name'].astype(str)]
            fig_ss = go.Figure()
            fig_ss.add_trace(go.Bar(x=wk['weekday_name'].astype(str), y=wk['ss_delta'], name='SS Added', marker_color=bar_colors))
            fig_ss.update_layout(title="Safety Stock Added (FC+SS − FC) by Day", height=300,
                margin=dict(l=40,r=20,t=40,b=40), yaxis_title="Units Added", plot_bgcolor='white',
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#f0f0f0'))
            st.plotly_chart(fig_ss, use_container_width=True)

        display_wk = wk[['weekday_name','qty_sold','forecast','forecast_ss','ss_delta','bias_pct','bias_ss_pct']].copy()
        display_wk.columns = ['Day','Qty Sold','FC','FC+SS','SS Added','Bias%[FC]','Bias%[FC+SS]']
        st.dataframe(
            display_wk.style
                .format({'Qty Sold':'{:,.0f}','FC':'{:,.0f}','FC+SS':'{:,.0f}',
                         'SS Added':'{:,.0f}','Bias%[FC]':'{:+.2f}%','Bias%[FC+SS]':'{:+.2f}%'})
                .apply(color_bias_col, subset=['Bias%[FC]','Bias%[FC+SS]'])
                .hide(axis='index'),
            use_container_width=True, height=290
        )
    else:
        st.info("ℹ️  No `plan_start_date` / `plan_end_date` columns found.")

    st.markdown("---")
    st.caption("Dashboard | Plan Forecast Bias Analysis | Data refreshes on upload")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — SAFETY STOCK DECOMPOSITION                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with tab_ss:

    st.markdown('<div class="section-header">🧮 Safety Stock Decomposition & Driver Analysis</div>', unsafe_allow_html=True)

    # Scope selector — entire filtered data OR drill into a subset
    scope_opt = st.radio(
        "Analyse scope",
        ["Entire filtered data", f"Subset: {groupby_dim} value"],
        horizontal=True, key="ss_scope"
    )
    if scope_opt.startswith("Subset"):
        dim_vals_ss = sorted(filtered[groupby_dim].dropna().unique().tolist()) if groupby_dim in filtered.columns else []
        ss_dim_val  = st.selectbox(f"Select {groupby_dim}", dim_vals_ss, key="ss_dim_val")
        ss_df = filtered[filtered[groupby_dim].astype(str) == str(ss_dim_val)].copy()
        scope_label = f"{groupby_dim} = {ss_dim_val}"
    else:
        ss_df = filtered.copy()
        scope_label = "All (filtered)"

    st.markdown(f"**Scope:** `{scope_label}`  — **{len(ss_df):,} plans**")
    st.markdown("---")

    # ── SS Component columns available ───────────────────────────────────────
    SS_COMPONENT_COLS = ['Z_SL','rmse_component','forecast_error_component',
                         'lead_time_variance_component','bias_adjustment',
                         'forecast_error_percent_bias','uncapped_safety_stock',
                         'new_ss_raw','new_ss_capped','original_safety_stock','average_sales_14day']
    avail_ss_cols = [c for c in SS_COMPONENT_COLS if c in ss_df.columns]

    # Flag columns
    flag_lt  = 'adjust_safety_stock_for_lead_time_variance' in ss_df.columns
    flag_bias= 'adjust_safety_stock_for_bias_error'         in ss_df.columns
    infl_col = next((c for c in ['infaltable','Infaltable','inflatable','Inflatable','is_inflatable']
                     if c in ss_df.columns), None)

    # ── 1. SS Component Averages — 2×2 matrix + delta contribution ───────────
    st.markdown("#### 1. SS Component Averages — by Segment")

    # resolve Z_SL column name (case-insensitive)
    _zsl_col = 'z_sl' if 'z_sl' in ss_df.columns else ('Z_SL' if 'Z_SL' in ss_df.columns else None)

    comp_cols_show = [c for c in [_zsl_col,'rmse_component','forecast_error_component',
                                   'lead_time_variance_component','bias_adjustment',
                                   'forecast_error_percent_bias','uncapped_safety_stock',
                                   'new_ss_raw','new_ss_capped','original_safety_stock']
                      if c is not None and c in ss_df.columns]

    # final SS output column
    ss_out_col = ('original_safety_stock' if 'original_safety_stock' in ss_df.columns
                  else ('new_ss_capped' if 'new_ss_capped' in ss_df.columns else None))

    # build segments
    segs = [("All", ss_df)]
    if infl_col:
        segs += [
            ("Inflatable",     ss_df[ss_df[infl_col] == True]),
            ("Non-Inflatable", ss_df[ss_df[infl_col] == False]),
        ]
        if 'xyz_class' in ss_df.columns:
            ni = ss_df[ss_df[infl_col] == False]
            segs += [
                ("Non-Infl / X+Y", ni[ni['xyz_class'].isin(['X','Y'])]),
                ("Non-Infl / Z",   ni[ni['xyz_class'] == 'Z']),
            ]

    # segment selector
    seg_names = [s[0] for s in segs if len(s[1]) > 0]
    sel_seg = st.selectbox("Segment to analyse", seg_names, key="seg_select")
    seg_data = next(d for n, d in segs if n == sel_seg).copy()

    # ── Delta-based contribution logic (proper isolation) ─────────────────────
    # SS = z_sl × rmse / bias_adj
    # where rmse = sqrt(forecast_error_component + lead_time_variance_component)
    #
    # Each delta = ss_actual − ss_with_component_set_to_neutral_value:
    #   delta_bias_adj  = ss_actual − (z_sl × rmse × 1)           bias_adj → 1
    #   delta_fe        = ss_actual − (z_sl × sqrt(lt) / bias_adj) fe → 0
    #   delta_lt        = ss_actual − (z_sl × sqrt(fe) / bias_adj) lt → 0
    #   delta_zsl       = ss_actual − (rmse / bias_adj)            z_sl → 1
    #
    # Contributions are |delta| / sum(|deltas|) × 100, so each answers:
    # "how much does SS drop if I remove just this component?"

    # required cols for contribution calc
    _required = [c for c in [_zsl_col,'rmse_component','forecast_error_component','bias_adjustment']
                 if c is not None]
    _missing   = [c for c in _required if c not in seg_data.columns]

    # also accept forecast_ss as fallback SS output
    if ss_out_col is None or ss_out_col not in seg_data.columns:
        ss_out_col = next((c for c in ['original_safety_stock','new_ss_capped',
                                       'uncapped_safety_stock','forecast_ss']
                           if c in seg_data.columns), None)

    has_comps = (len(_missing) == 0) and (ss_out_col is not None)

    if not has_comps:
        st.warning(f"Cannot compute contributions. Missing columns: {_missing}. "
                   f"SS output col found: {ss_out_col}")

    if has_comps:
        d = seg_data.copy()
        eps = 1e-9

        ba  = d['bias_adjustment'].replace(0, eps)
        fe  = d['forecast_error_component'].clip(lower=0)
        lt  = d['lead_time_variance_component'].clip(lower=0) if 'lead_time_variance_component' in d.columns else pd.Series(0, index=d.index)
        zsl = d[_zsl_col]
        rmse = d['rmse_component']

        # SS = z_sl × rmse / bias_adj  (actual formula)
        d['_ss_actual'] = d[ss_out_col]

        # ── Proper isolation deltas ────────────────────────────────────────────
        # Each delta = ss_actual - ss_with_that_component_removed
        # "removed" means: set to its neutral value (1 for bias_adj, 0 for variance terms)

        # delta_bias_adj: set bias_adj = 1  →  ss = z × rmse
        d['_ss_no_bias']    = zsl * rmse                              # bias_adj → 1
        d['delta_bias_adj'] = d['_ss_actual'] - d['_ss_no_bias']     # negative means bias_adj inflates SS

        # delta_fe: remove forecast_error_component  →  rmse = sqrt(lt)
        rmse_no_fe = np.sqrt(lt)
        d['_ss_no_fe']  = zsl * rmse_no_fe / ba
        d['delta_fe']   = d['_ss_actual'] - d['_ss_no_fe']

        # delta_lt: remove lead_time_variance_component  →  rmse = sqrt(fe)
        rmse_no_lt = np.sqrt(fe)
        d['_ss_no_lt']  = zsl * rmse_no_lt / ba
        d['delta_lt']   = d['_ss_actual'] - d['_ss_no_lt']

        # delta_zsl: set z_sl = 1  →  ss = rmse / bias_adj
        d['_ss_no_zsl'] = rmse / ba
        d['delta_zsl']  = d['_ss_actual'] - d['_ss_no_zsl']

        # avg deltas (use absolute value — we care about magnitude of impact)
        avg_d_bias = d['delta_bias_adj'].mean()
        avg_d_fe   = d['delta_fe'].mean()
        avg_d_lt   = d['delta_lt'].mean()
        avg_d_zsl  = d['delta_zsl'].mean()

        # normalise to 100% using absolute values so sign doesn't cancel contributions
        raw_contribs = {
            'bias_adjustment':             abs(avg_d_bias),
            'forecast_error_component':    abs(avg_d_fe),
            'lead_time_variance_component':abs(avg_d_lt),
            _zsl_col:                      abs(avg_d_zsl),
        }
        total_contrib = sum(raw_contribs.values()) + eps
        norm_pct = {k: 100 * v / total_contrib for k, v in raw_contribs.items()}
    else:
        norm_pct = {}

    # ── 2×2 grid ─────────────────────────────────────────────────────────────
    def cell_html(title, avg_val, contrib_pct, color):
        bar_w = min(int(contrib_pct), 100)
        return f"""
        <div style="border:1px solid #dee2e6; border-radius:8px; padding:14px 16px;
                    background:#fafafa; height:140px;">
            <div style="font-size:11px;color:#6c757d;font-weight:700;
                        text-transform:uppercase;letter-spacing:.5px;">{title}</div>
            <div style="font-size:20px;font-weight:700;color:#212529;margin:4px 0;">
                {avg_val:.4f}</div>
            <div style="font-size:11px;color:#495057;margin-bottom:4px;">
                Contribution: <b style="color:{color};">{contrib_pct:.1f}%</b></div>
            <div style="background:#e9ecef;border-radius:4px;height:6px;">
                <div style="width:{bar_w}%;background:{color};height:6px;
                            border-radius:4px;"></div>
            </div>
        </div>"""

    grid_r1c1, grid_r1c2 = st.columns(2)
    grid_r2c1, grid_r2c2 = st.columns(2)

    ba_avg  = seg_data['bias_adjustment'].mean()               if 'bias_adjustment'              in seg_data.columns else 0
    lt_avg  = seg_data['lead_time_variance_component'].mean()  if 'lead_time_variance_component'  in seg_data.columns else 0
    fe_avg  = seg_data['forecast_error_component'].mean()      if 'forecast_error_component'       in seg_data.columns else 0
    zsl_avg = seg_data[_zsl_col].mean()                        if _zsl_col and _zsl_col            in seg_data.columns else 0

    with grid_r1c1:
        st.markdown(cell_html("bias_adjustment",              ba_avg,  norm_pct.get('bias_adjustment',0),              "#e53e3e"), unsafe_allow_html=True)
    with grid_r1c2:
        st.markdown(cell_html("lead_time_variance_component", lt_avg,  norm_pct.get('lead_time_variance_component',0), "#dd6b20"), unsafe_allow_html=True)
    with grid_r2c1:
        st.markdown(cell_html("forecast_error_component",     fe_avg,  norm_pct.get('forecast_error_component',0),     "#2b6cb0"), unsafe_allow_html=True)
    with grid_r2c2:
        st.markdown(cell_html(_zsl_col or "z_sl",             zsl_avg, norm_pct.get(_zsl_col or 'z_sl', 0),           "#276749"), unsafe_allow_html=True)

    # raw data expander — plan-level with all component cols
    raw_keep = [c for c in ['plan_index','storereferenceid','warehouseid',
                'forecast','forecast_ss','quantity_sold'] + comp_cols_show if c in seg_data.columns]
    raw_all = seg_data[raw_keep].copy()
    with st.expander("📋 View raw plan-level data", expanded=False):
        st.dataframe(raw_all, use_container_width=True, height=350)
        csv = raw_all.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download as CSV", data=csv,
                           file_name="ss_segment_raw.csv", mime="text/csv",
                           key="dl_ss_seg_raw")

    st.markdown("---")

    # ── 3. SS Driver Analysis — by groupby_dim ────────────────────────────────
    st.markdown(f"#### 3. SS Driver Analysis — grouped by `{groupby_dim}`")

    if groupby_dim in ss_df.columns and comp_cols_show and ss_out_col:
        # SS share% = mean(SS / order_qty) at row level, then aggregate
        drv_df = ss_df.copy()
        drv_df['ss_pct_of_oq'] = (drv_df[ss_out_col] /
                                   drv_df['quantity_sold'].replace(0, np.nan) * 100)

        driver_agg = (drv_df.groupby(groupby_dim)
                      .agg(avg_SS_pct_of_OQ=('ss_pct_of_oq', 'mean'),
                           total_SS=(ss_out_col, 'sum'),
                           N=('ss_pct_of_oq', 'count'))
                      .reset_index())
        comp_means = drv_df.groupby(groupby_dim)[comp_cols_show].mean().reset_index()
        driver_agg = driver_agg.merge(comp_means, on=groupby_dim, how='left')
        driver_agg = driver_agg.sort_values('avg_SS_pct_of_OQ', ascending=False)

        # determine which component column is highest per row → colour it
        driver_input_cols = [c for c in
                             ['rmse_component','forecast_error_component',
                              'lead_time_variance_component','bias_adjustment']
                             if c in driver_agg.columns]

        fig_drv = go.Figure(go.Bar(
            x=driver_agg[groupby_dim].astype(str),
            y=driver_agg['avg_SS_pct_of_OQ'],
            marker_color='steelblue', opacity=0.85,
            text=driver_agg['avg_SS_pct_of_OQ'].map('{:.1f}%'.format),
            textposition='outside'
        ))
        fig_drv.update_layout(
            title=f"Avg SS% of Order Qty by {groupby_dim}", height=300,
            margin=dict(l=40,r=20,t=40,b=40), yaxis_title="Avg SS% of OQ",
            plot_bgcolor='white', xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig_drv, use_container_width=True)

        # table: colour ss_out_col red, plus highlight max driver component col per row
        def style_driver_table(df):
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            # colour the SS output col always red-gradient
            if ss_out_col in df.columns:
                max_ss = df[ss_out_col].max()
                for i, val in enumerate(df[ss_out_col]):
                    if pd.isna(val) or not max_ss:
                        intensity = 255
                    else:
                        intensity = max(0, min(255, int(255 - 160 * (val / max_ss))))
                    styles.iloc[i, df.columns.get_loc(ss_out_col)] = (
                        f'background-color: rgb(255,{intensity},{intensity}); font-weight:600')
            # highlight the biggest input driver col per row
            if driver_input_cols:
                for i, row in df[driver_input_cols].iterrows():
                    if row.notna().any():
                        max_c = row.idxmax()
                        styles.iloc[i, df.columns.get_loc(max_c)] = (
                            'background-color: #ffe0e0; color: #c0392b; font-weight:600')
            return styles

        fmt = {'N':'{:,}','avg_SS_pct_of_OQ':'{:.2f}%','total_SS':'{:,.2f}'}
        for c in comp_cols_show:
            fmt[c] = '{:.3f}'

        st.dataframe(
            driver_agg.style.apply(style_driver_table, axis=None).format(fmt).hide(axis='index'),
            use_container_width=True, height=300
        )

    st.markdown("---")

    # ── 4. Component Distributions ────────────────────────────────────────────
    st.markdown("#### 4. Component Distributions")

    dist_cols = [c for c in [_zsl_col,'rmse_component','forecast_error_component',
                              'lead_time_variance_component','bias_adjustment',
                              'forecast_error_percent_bias'] if c and c in ss_df.columns]

    if dist_cols:
        selected_dist_col = st.selectbox("Select component to inspect", dist_cols, key="dist_col")
        dist_data = ss_df[selected_dist_col].dropna()
        p95 = float(dist_data.quantile(0.95))

        dcol1, dcol2 = st.columns([2, 1])
        with dcol1:
            plot_data = ss_df[ss_df[selected_dist_col] <= p95].dropna(subset=[selected_dist_col])
            if groupby_dim in ss_df.columns and ss_df[groupby_dim].nunique() <= 10:
                fig_hist = px.histogram(
                    plot_data, x=selected_dist_col, color=groupby_dim,
                    nbins=50, barmode='overlay', opacity=0.7,
                    title=f"Distribution of {selected_dist_col} by {groupby_dim} (capped at p95)"
                )
            else:
                fig_hist = px.histogram(
                    plot_data, x=selected_dist_col, nbins=50,
                    title=f"Distribution of {selected_dist_col} (capped at p95)"
                )
            fig_hist.update_layout(height=320, margin=dict(l=40,r=20,t=40,b=40), plot_bgcolor='white')
            st.plotly_chart(fig_hist, use_container_width=True)

        with dcol2:
            pcts = dist_data.quantile([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).reset_index()
            pcts.columns = ['Percentile','Value']
            pcts['Percentile'] = pcts['Percentile'].map(lambda x: f"p{int(x*100)}")
            st.markdown(f"**{selected_dist_col} — percentiles**")
            st.dataframe(pcts.style.format({'Value':'{:.4f}'}).hide(axis='index'),
                         use_container_width=True, height=280)

    st.markdown("---")

    # ── 5. Bias Adjustment Impact ─────────────────────────────────────────────
    st.markdown("#### 5. Bias Adjustment Impact")

    # let user pick which bias column to use on X axis
    bias_x_options = [c for c in ['forecast_error_percent_bias','BT_forecast_error_percent_bias']
                      if c in ss_df.columns]

    if 'bias_adjustment' in ss_df.columns and bias_x_options:
        selected_bias_x = st.selectbox("X-axis bias column", bias_x_options, key="bias_x_col")

        ba_col1, ba_col2 = st.columns(2)
        with ba_col1:
            _ba_pool = ss_df.dropna(subset=[selected_bias_x, 'bias_adjustment'])
            ba_plot  = _ba_pool.sample(min(3000, len(_ba_pool)), random_state=42)
            fig_ba = px.scatter(
                ba_plot, x=selected_bias_x, y='bias_adjustment',
                color=groupby_dim if groupby_dim in ss_df.columns else None,
                opacity=0.4,
                title=f"{selected_bias_x} vs bias_adjustment"
            )
            fig_ba.update_layout(height=320, margin=dict(l=40,r=20,t=40,b=40), plot_bgcolor='white')
            st.plotly_chart(fig_ba, use_container_width=True)

        with ba_col2:
            if groupby_dim in ss_df.columns:
                agg_dict = {
                    'bias_adjustment': 'mean',
                    selected_bias_x:   'mean',
                }
                if flag_bias:
                    agg_dict['adjust_safety_stock_for_bias_error'] = 'mean'
                ba_grp = ss_df.groupby(groupby_dim).agg(agg_dict).reset_index()
                ba_grp.columns = (
                    [groupby_dim, 'avg_bias_adj', f'avg_{selected_bias_x}']
                    + (['pct_bias_adjusted'] if flag_bias else [])
                )
                fmt_ba = {'avg_bias_adj':'{:.3f}', f'avg_{selected_bias_x}':'{:.3f}'}
                if flag_bias:
                    fmt_ba['pct_bias_adjusted'] = '{:.1%}'
                st.markdown(f"**Avg bias adjustment by {groupby_dim}**")
                st.dataframe(ba_grp.style.format(fmt_ba).hide(axis='index'),
                             use_container_width=True, height=250)
    else:
        st.info("bias_adjustment or forecast_error_percent_bias column not found.")

    st.markdown("---")
    st.caption("Safety Stock Decomposition | Data refreshes on upload")