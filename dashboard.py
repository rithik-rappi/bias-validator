import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Forecast Bias Dashboard", page_icon="📦")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .metric-box-red {
        background: #fff5f5;
        border: 1px solid #fc8181;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .kpi-label { font-size: 11px; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
    .kpi-value { font-size: 22px; font-weight: 700; color: #212529; }
    .kpi-value-red { font-size: 22px; font-weight: 700; color: #e53e3e; }
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

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)
    # parse date columns
    for _dc in ['plan_start_date', 'plan_end_date']:
        if _dc in df.columns:
            df[_dc] = pd.to_datetime(df[_dc], errors='coerce')
    # ensure numeric
    for col in ['quantity_sold','forecast','forecast_ss','error_capped','bias',
                'swa','doi','error_ss_capped','bias_ss','availability','bias%','bias_ss%']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

uploaded = st.file_uploader("Upload plan_data parquet file", type=["parquet"])
if uploaded is None:
    st.info("⬆️  Upload the parquet file to begin.")
    st.stop()

plan_data = load_data(uploaded)

# ── Sidebar – Global filters ───────────────────────────────────────────────────
st.sidebar.header("🔧 Global Filters")

# qty_sold > 0 checkbox
filter_qty = st.sidebar.checkbox("Filter: quantity_sold > 0", value=False)
if filter_qty:
    plan_data = plan_data[plan_data['quantity_sold'] > 0]

# availability >= 85% checkbox
filter_avb = st.sidebar.checkbox("Filter: availability ≥ 85%", value=False)
if filter_avb and 'availability' in plan_data.columns:
    plan_data = plan_data[plan_data['availability'] >= 85]

FILTER_COLS = ['sku_classification','time_series_class_day','category_name',
               'city_name','abc_class','xyz_class','coverage_bucket','coverage_bucket_ss']

active_filters = {}
for col in FILTER_COLS:
    if col in plan_data.columns:
        opts = sorted(plan_data[col].dropna().unique().tolist())
        sel = st.sidebar.multiselect(col, opts, default=[], key=f"filter_{col}")
        if sel:
            active_filters[col] = sel

# Apply global filters
filtered = plan_data.copy()
for col, vals in active_filters.items():
    filtered = filtered[filtered[col].isin(vals)]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Rows after filters:** `{len(filtered):,}`")

# ── Groupby selector ──────────────────────────────────────────────────────────
DIM_COLS = [c for c in FILTER_COLS if c in plan_data.columns]
groupby_dim = st.sidebar.selectbox("📐 Group-by dimension", DIM_COLS, index=0)

# ── Helper functions ───────────────────────────────────────────────────────────
def compute_metrics(df):
    qs  = df['quantity_sold'].sum()
    fc  = df['forecast'].sum()
    fss = df['forecast_ss'].sum()
    wmape_fc  = 100 * df['error_capped'].sum() / qs  if qs else np.nan
    bias_fc   = 100 * df['bias'].sum() / qs           if qs else np.nan
    swa_fc    = df['swa'].mean()                       if 'swa' in df.columns else np.nan
    doi_fc    = df['doi'].mean()                       if 'doi' in df.columns else np.nan
    wmape_ss  = 100 * df['error_ss_capped'].sum() / qs if qs else np.nan
    bias_ss   = 100 * df['bias_ss'].sum() / qs         if qs else np.nan
    return dict(qty_sold=qs, fc=fc, fc_ss=fss,
                wmape_fc=wmape_fc, bias_fc=bias_fc, swa_fc=swa_fc, doi_fc=doi_fc,
                wmape_ss=wmape_ss, bias_ss=bias_ss)

def make_metrics_table(df):
    avb_high = df[df['availability'] >= 85] if 'availability' in df.columns else df
    avb_low  = df[df['availability'] <  85] if 'availability' in df.columns else df.iloc[0:0]
    rows = []
    for label, subset in [("All", df), ("Avb ≥ 85%", avb_high), ("Avb < 85%", avb_low)]:
        m = compute_metrics(subset)
        rows.append({
            "Segment": label,
            "Filters": subset['plan_index'].nunique() if 'plan_index' in subset.columns else len(subset),
            "Qty Sold": int(m['qty_sold']),
            "FC": int(m['fc']),
            "FC+SS": int(m['fc_ss']),
            "wMAPE [FC]": f"{m['wmape_fc']:.2f}%",
            "Bias [FC]": m['bias_fc'],
            "SWA": f"{m['swa_fc']:.2f}%" if not np.isnan(m['swa_fc']) else "—",
            "DOI": f"{m['doi_fc']:.2f}" if not np.isnan(m['doi_fc']) else "—",
            "wMAPE [FC+SS]": f"{m['wmape_ss']:.2f}%",
            "Bias [FC+SS]": m['bias_ss'],
        })
    return pd.DataFrame(rows)

def style_bias(val):
    if isinstance(val, float):
        color = "#e53e3e" if val > 0 else "#2b6cb0"
        return f"color: {color}; font-weight: 600;"
    return ""

def of_analysis(df, dim):
    """Consistent-overforecast analysis grouped by `dim`."""
    needed = ['plan_index','storereferenceid','warehouseid', dim, 'bias%','bias_ss%','bias','bias_ss']
    plan_df = df[[c for c in needed if c in df.columns]].copy()

    if 'storereferenceid' not in plan_df.columns or 'warehouseid' not in plan_df.columns:
        return pd.DataFrame(), plan_df, pd.DataFrame()

    plan_count = plan_df.groupby(['storereferenceid','warehouseid']).size().reset_index(name='plan_count')
    plan_df = plan_df.merge(plan_count, on=['storereferenceid','warehouseid'], how='left')

    pstv = plan_df[plan_df['bias%'] > 0].groupby(['storereferenceid','warehouseid']).size().reset_index(name='pstv_FC_bias_count')
    plan_df = plan_df.merge(pstv, on=['storereferenceid','warehouseid'], how='left')
    plan_df['pstv_FC_bias_count'] = plan_df['pstv_FC_bias_count'].fillna(0)
    plan_df['pstv_plans_ratio'] = plan_df['pstv_FC_bias_count'] / plan_df['plan_count']

    consistent_OF = plan_df[plan_df['pstv_plans_ratio'] > 0.8]

    OF_filters = (
        plan_df.groupby(dim).size().reset_index(name='plan-filter_combinations')
        .merge(consistent_OF.groupby(dim).size().reset_index(name='OF_plan-filter_combinations'),
               on=dim, how='left')
    )
    OF_filters['OF_plan-filter_combinations'] = OF_filters['OF_plan-filter_combinations'].fillna(0).astype(int)
    OF_filters['filter%'] = (OF_filters['OF_plan-filter_combinations'] /
                              OF_filters['plan-filter_combinations'] * 100).round(1)

    OF_bias = (
        plan_df.groupby(dim)['bias'].sum().reset_index(name='plan-filter_bias')
        .merge(consistent_OF.groupby(dim)['bias'].sum().reset_index(name='OF_plan-filter_bias'),
               on=dim, how='left')
    )
    OF_bias['OF_plan-filter_bias'] = OF_bias['OF_plan-filter_bias'].fillna(0).round(0).astype(int)
    OF_bias['plan-filter_bias']    = OF_bias['plan-filter_bias'].round(0).astype(int)
    OF_bias['bias_contri%'] = (OF_bias['OF_plan-filter_bias'] /
                                OF_bias['plan-filter_bias'].replace(0, np.nan) * 100).round(1)

    result = OF_filters.merge(OF_bias, on=dim, how='left')
    return result, plan_df, consistent_OF

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📦 Forecast Bias & Safety Stock Dashboard")

# ── TOP ROW: KPI boxes ─────────────────────────────────────────────────────────
qs_total = filtered['quantity_sold'].sum()
bias_agg   = 100 * filtered['bias'].sum()   / qs_total if qs_total else 0
bias_ss_agg= 100 * filtered['bias_ss'].sum()/ qs_total if qs_total else 0
ss_added   = filtered['forecast_ss'].sum() - filtered['forecast'].sum()

col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)

with col_k1:
    st.markdown(f"""<div class="metric-box">
        <div class="kpi-label">Total Qty Sold</div>
        <div class="kpi-value">{int(qs_total):,}</div></div>""", unsafe_allow_html=True)

with col_k2:
    st.markdown(f"""<div class="metric-box">
        <div class="kpi-label">Total Forecast</div>
        <div class="kpi-value">{int(filtered['forecast'].sum()):,}</div></div>""", unsafe_allow_html=True)

with col_k3:
    st.markdown(f"""<div class="metric-box">
        <div class="kpi-label">Total FC+SS</div>
        <div class="kpi-value">{int(filtered['forecast_ss'].sum()):,}</div></div>""", unsafe_allow_html=True)

with col_k4:
    color_cls = "kpi-value-red" if bias_agg > 0 else "kpi-value"
    st.markdown(f"""<div class="metric-box-red">
        <div class="kpi-label">Bias% [FC] (wt. avg)</div>
        <div class="{color_cls}">{bias_agg:+.2f}%</div></div>""", unsafe_allow_html=True)

with col_k5:
    color_cls2 = "kpi-value-red" if bias_ss_agg > 0 else "kpi-value"
    st.markdown(f"""<div class="metric-box-red">
        <div class="kpi-label">Bias% [FC+SS] (wt. avg)</div>
        <div class="{color_cls2}">{bias_ss_agg:+.2f}%</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# ── SECTION 1: Metrics Table ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Accuracy Metrics by Availability</div>', unsafe_allow_html=True)

metrics_df = make_metrics_table(filtered)

def color_bias_col(s):
    return [style_bias(v) for v in s]

styled = (metrics_df.style
    .format({"Qty Sold": "{:,}", "FC": "{:,}", "FC+SS": "{:,}",
             "Bias [FC]": "{:+.2f}%", "Bias [FC+SS]": "{:+.2f}%"})
    .apply(color_bias_col, subset=["Bias [FC]", "Bias [FC+SS]"])
    .set_properties(**{'text-align': 'right'}, subset=["Qty Sold","FC","FC+SS"])
    .set_properties(**{'font-weight': '600'}, subset=["Segment"])
    .hide(axis='index')
)
st.dataframe(styled, use_container_width=True, height=160)

st.markdown("---")

# ── SECTION 2: OF Analysis + Quadrant Plot ─────────────────────────────────────
st.markdown(f'<div class="section-header">🔍 Over-Forecast Analysis by {groupby_dim}</div>', unsafe_allow_html=True)

try:
    of_result, plan_df_full, consistent_OF_df = of_analysis(filtered, groupby_dim)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"**Consistent OF (>80% positive bias plans) — grouped by `{groupby_dim}`**")

        def style_of_table(df):
            fmt = {'filter%': '{:.1f}%', 'bias_contri%': '{:.1f}%',
                   'plan-filter_bias': '{:,}', 'OF_plan-filter_bias': '{:,}'}
            s = df.style.format(fmt).hide(axis='index')
            if 'filter%' in df.columns:
                s = s.background_gradient(cmap='RdYlGn_r', subset=['filter%'], vmin=0, vmax=100)
            if 'bias_contri%' in df.columns:
                s = s.background_gradient(cmap='RdYlGn_r', subset=['bias_contri%'], vmin=0, vmax=100)
            return s

        if not of_result.empty:
            st.dataframe(style_of_table(of_result), use_container_width=True, height=350)
        else:
            st.warning("OF analysis requires storereferenceid / warehouseid columns.")

        # Dimension value selector
        dim_values = sorted(filtered[groupby_dim].dropna().unique().tolist()) if groupby_dim in filtered.columns else []
        selected_dim_val = st.selectbox(
            f"🔎 Select {groupby_dim} value to drill down",
            ["(All)"] + [str(v) for v in dim_values],
            key="dim_drill"
        )

    with col_right:
        st.markdown(f"**Bias% vs Bias_SS% — 4-Quadrant Scatter**")

        # Build scatter data
        scatter_df = filtered.copy()
        if 'bias%' in scatter_df.columns and 'bias_ss%' in scatter_df.columns:
            scatter_df = scatter_df.dropna(subset=['bias%','bias_ss%'])

            # Filter to selected dim value if applicable
            if selected_dim_val != "(All)" and groupby_dim in scatter_df.columns:
                scatter_plot_df = scatter_df[scatter_df[groupby_dim].astype(str) == selected_dim_val]
            else:
                scatter_plot_df = scatter_df

            # Sample for performance
            MAX_POINTS = 5000
            if len(scatter_plot_df) > MAX_POINTS:
                scatter_plot_df = scatter_plot_df.sample(MAX_POINTS, random_state=42)

            if len(scatter_plot_df) == 0:
                st.warning("No data to plot for this selection.")
            else:
                x_range = scatter_plot_df['bias%'].quantile([0.01,0.99]).values
                y_range = scatter_plot_df['bias_ss%'].quantile([0.01,0.99]).values
                # Guard against degenerate ranges
                x_lo = min(x_range[0], 0) * 1.1 - 1
                x_hi = max(x_range[1], 0) * 1.1 + 1
                y_lo = min(y_range[0], 0) * 1.1 - 1
                y_hi = max(y_range[1], 0) * 1.1 + 1

                fig_q = go.Figure()

                # Quadrant shading
                for (x0,x1,y0,y1,color,label) in [
                    (0, x_hi, 0, y_hi, 'rgba(252,129,129,0.10)', 'OF / OF+SS'),
                    (x_lo, 0, 0, y_hi, 'rgba(107,194,107,0.10)', 'UF / OF+SS'),
                    (0, x_hi, y_lo, 0, 'rgba(255,166,77,0.10)', 'OF / UF+SS'),
                    (x_lo, 0, y_lo, 0, 'rgba(100,149,237,0.10)', 'UF / UF+SS'),
                ]:
                    fig_q.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                                   fillcolor=color, line_width=0, layer="below")
                    fig_q.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=label,
                                        showarrow=False, font=dict(size=9, color='#888'),
                                        opacity=0.7)

                # Reference lines
                fig_q.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                fig_q.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)

                # Color by groupby_dim
                if groupby_dim in scatter_plot_df.columns and selected_dim_val == "(All)":
                    for cat_val in scatter_plot_df[groupby_dim].unique():
                        sub = scatter_plot_df[scatter_plot_df[groupby_dim] == cat_val]
                        hover_cols = [groupby_dim]
                        for hc in ['storereferenceid','warehouseid']:
                            if hc in sub.columns:
                                hover_cols.append(hc)
                        hover_text = sub.apply(
                            lambda r: "<br>".join(
                                [f"{c}: {r[c]}" for c in hover_cols if c in r.index] +
                                [f"bias%: {r['bias%']:.2f}%", f"bias_ss%: {r['bias_ss%']:.2f}%"]),
                            axis=1
                        )
                        fig_q.add_trace(go.Scatter(
                            x=sub['bias%'], y=sub['bias_ss%'],
                            mode='markers', name=str(cat_val),
                            text=hover_text, hoverinfo='text',
                            marker=dict(size=4, opacity=0.55)
                        ))
                else:
                    fig_q.add_trace(go.Scatter(
                        x=scatter_plot_df['bias%'], y=scatter_plot_df['bias_ss%'],
                        mode='markers', name=str(selected_dim_val),
                        marker=dict(size=4, opacity=0.55, color='steelblue')
                    ))

                fig_q.update_layout(
                    xaxis_title="Bias% [FC]", yaxis_title="Bias% [FC+SS]",
                    height=380, margin=dict(l=40,r=20,t=20,b=40),
                    legend=dict(font=dict(size=10), itemsizing='constant'),
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False, range=[x_lo, x_hi]),
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=False, range=[y_lo, y_hi]),
                )
                st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.warning("Columns `bias%` / `bias_ss%` not found in data.")

    # ── Drilldown section ─────────────────────────────────────────────────────
    if selected_dim_val != "(All)":
        st.markdown("---")
        st.markdown(f'<div class="section-header">🔬 Drill-down: {groupby_dim} = "{selected_dim_val}"</div>',
                    unsafe_allow_html=True)

        drill_df = filtered[filtered[groupby_dim].astype(str) == selected_dim_val]

        # KPI box for subset — top right
        qs_d = drill_df['quantity_sold'].sum()
        b_d  = 100 * drill_df['bias'].sum()    / qs_d if qs_d else 0
        bs_d = 100 * drill_df['bias_ss'].sum() / qs_d if qs_d else 0

        _, kd1, kd2, kd3 = st.columns([3,1,1,1])
        with kd1:
            st.markdown(f"""<div class="metric-box"><div class="kpi-label">Qty Sold (subset)</div>
                <div class="kpi-value">{int(qs_d):,}</div></div>""", unsafe_allow_html=True)
        with kd2:
            cc = "kpi-value-red" if b_d > 0 else "kpi-value"
            st.markdown(f"""<div class="metric-box-red"><div class="kpi-label">Bias% [FC]</div>
                <div class="{cc}">{b_d:+.2f}%</div></div>""", unsafe_allow_html=True)
        with kd3:
            cc2 = "kpi-value-red" if bs_d > 0 else "kpi-value"
            st.markdown(f"""<div class="metric-box-red"><div class="kpi-label">Bias% [FC+SS]</div>
                <div class="{cc2}">{bs_d:+.2f}%</div></div>""", unsafe_allow_html=True)

        # Coverage matrix (if available)
        if 'coverage_bucket' in filtered.columns and 'coverage_bucket_ss' in filtered.columns:
            matrix = pd.crosstab(drill_df['coverage_bucket'], drill_df['coverage_bucket_ss'])
            st.markdown("**Coverage Bucket Matrix** (rows = FC coverage, cols = FC+SS coverage)")
            st.dataframe(matrix.style.background_gradient(cmap='YlOrRd').format("{:,}"),
                         use_container_width=True)

        # Filtered metrics table for subset
        st.markdown(f"**Accuracy metrics for {groupby_dim} = {selected_dim_val}**")
        sub_metrics = make_metrics_table(drill_df)
        st.dataframe(
            sub_metrics.style
                .format({"Qty Sold": "{:,}", "FC": "{:,}", "FC+SS": "{:,}",
                         "Bias [FC]": "{:+.2f}%", "Bias [FC+SS]": "{:+.2f}%"})
                .apply(color_bias_col, subset=["Bias [FC]", "Bias [FC+SS]"])
                .hide(axis='index'),
            use_container_width=True, height=160
        )

except KeyError as e:
    st.error(f"Column not found: {e}. Check that required columns exist in the data.")
except Exception as e:
    st.error(f"Error in OF analysis: {e}")

st.markdown("---")

# ── SECTION 3: Weekend / Weekday Pattern ──────────────────────────────────────
_has_date_src = any(c in filtered.columns for c in ['plan_end_date','plan_start_date'])
if _has_date_src:
    st.markdown('<div class="section-header">📅 Weekend / Weekday Safety Stock Pattern</div>',
                unsafe_allow_html=True)

    wd_df = filtered.copy()

    # -- Explode each plan into one row per day of its duration ----------------
    wd_df['plan_start_date'] = pd.to_datetime(wd_df['plan_start_date'], errors='coerce')
    wd_df['plan_end_date']   = pd.to_datetime(wd_df['plan_end_date'],   errors='coerce')

    wd_df['date'] = wd_df.apply(
        lambda r: pd.date_range(r['plan_start_date'], r['plan_end_date'], freq='D')
        if pd.notna(r['plan_start_date']) and pd.notna(r['plan_end_date']) else [],
        axis=1
    )
    wd_df = wd_df.explode('date').reset_index(drop=True)

    # -- Scale numeric cols by duration so totals stay correct ----------------
    wd_df['duration_days'] = (
        wd_df['plan_end_date'] - wd_df['plan_start_date']
    ).dt.days + 1

    for _col in ['quantity_sold','forecast','forecast_ss','bias','bias_ss',
                 'error_capped','error_ss_capped']:
        if _col in wd_df.columns:
            wd_df[_col] = wd_df[_col] / wd_df['duration_days']

    # -- DOW from exploded date ------------------------------------------------
    wd_df['weekday_name'] = wd_df['date'].dt.day_name()
    wd_df['is_weekend']   = wd_df['date'].dt.dayofweek >= 5

    DOW_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    wk = (wd_df.groupby('weekday_name')
          .agg(qty_sold=('quantity_sold','sum'),
               forecast=('forecast','sum'),
               forecast_ss=('forecast_ss','sum'),
               bias_sum=('bias','sum'),
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
        fig_wk.add_trace(go.Bar(x=wk['weekday_name'].astype(str), y=wk['bias_pct'],
                                name='Bias% [FC]', marker_color='steelblue', opacity=0.8))
        fig_wk.add_trace(go.Bar(x=wk['weekday_name'].astype(str), y=wk['bias_ss_pct'],
                                name='Bias% [FC+SS]', marker_color='tomato', opacity=0.8))
        fig_wk.update_layout(
            title="Bias% by Day of Week",
            barmode='group', height=300, margin=dict(l=40,r=20,t=40,b=40),
            yaxis_title="Bias%", plot_bgcolor='white',
            legend=dict(font=dict(size=10)),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', zeroline=True, zerolinecolor='#aaa'),
        )
        st.plotly_chart(fig_wk, use_container_width=True)

    with wk_col2:
        bar_colors = [
            'rgba(255,140,60,0.85)' if d in ['Saturday','Sunday']
            else 'rgba(80,160,200,0.75)'
            for d in wk['weekday_name'].astype(str)
        ]
        fig_ss = go.Figure()
        fig_ss.add_trace(go.Bar(x=wk['weekday_name'].astype(str), y=wk['ss_delta'],
                                name='SS Added', marker_color=bar_colors))
        fig_ss.update_layout(
            title="Safety Stock Added (FC+SS − FC) by Day",
            height=300, margin=dict(l=40,r=20,t=40,b=40),
            yaxis_title="Units Added", plot_bgcolor='white',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
        )
        st.plotly_chart(fig_ss, use_container_width=True)

    # Summary table
    display_wk = wk[['weekday_name','qty_sold','forecast','forecast_ss',
                      'ss_delta','bias_pct','bias_ss_pct']].copy()
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
    st.info("ℹ️  No `plan_start_date` / `plan_end_date` columns found — weekend/weekday pattern analysis unavailable.")

st.markdown("---")
st.caption("Dashboard | Plan Forecast Bias Analysis | Data refreshes on upload")