import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

INK = "#1E1B3A"
MUTED = "#6B6785"
LINE = "#E6E2F2"
VIOLET = "#5B3FD9"
MAGENTA = "#D6246E"
ORANGE = "#FF8A3D"
TEAL = "#2BB3A3"
PALETTE = [VIOLET, MAGENTA, ORANGE, TEAL, "#F2B134", "#8B7CF0"]

# Diverging (correlations) and sequential (heatmaps) color maps
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "brand_div", [VIOLET, "#FFFFFF", MAGENTA]
)
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "brand_seq", ["#F4F0FF", "#A58CF0", VIOLET, MAGENTA, ORANGE]
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #1E1B3A;
  --muted: #6B6785;
  --line: #E6E2F2;
  --violet: #5B3FD9;
  --magenta: #D6246E;
  --orange: #FF8A3D;
}

.stApp { background: #F6F4FB; color: var(--ink); }
.stApp, .stMarkdown, p, label, input, button, textarea, li, td, th {
  font-family: 'DM Sans', system-ui, sans-serif;
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 {
  font-family: 'Bricolage Grotesque', 'DM Sans', sans-serif !important;
  color: var(--ink);
  letter-spacing: -0.02em;
}
h3 { font-weight: 800; }

/* Hero */
.hero {
  background: linear-gradient(120deg, #3B2BA8 0%, #7A2FC0 38%, #D6246E 72%, #FF8A3D 100%);
  border-radius: 28px;
  padding: 2.8rem 3rem 2.4rem 3rem;
  color: #fff;
  position: relative;
  overflow: hidden;
  margin-bottom: 0.6rem;
}
.hero::after {
  content: "";
  position: absolute;
  right: -90px; top: -90px;
  width: 340px; height: 340px;
  border-radius: 50%;
  border: 38px solid rgba(255,255,255,0.10);
}
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  font-weight: 800;
  font-size: 3rem;
  line-height: 1.05;
  color: #fff !important;
  margin: 0 0 0.8rem 0;
  padding: 0;
  max-width: 760px;
}
.hero p {
  font-size: 1.12rem;
  line-height: 1.55;
  max-width: 640px;
  color: rgba(255,255,255,0.92);
  margin: 0 0 1.4rem 0;
}
.hero .chips { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero .chip {
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.35);
  border-radius: 999px;
  padding: 0.35rem 0.95rem;
  font-size: 0.92rem;
  font-weight: 500;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.1rem 1.3rem 1rem 1.3rem;
  box-shadow: 0 6px 20px rgba(60, 40, 140, 0.06);
  border-top: 4px solid var(--violet);
}
[data-testid="stColumn"]:nth-of-type(2) [data-testid="stMetric"] { border-top-color: var(--magenta); }
[data-testid="stColumn"]:nth-of-type(3) [data-testid="stMetric"] { border-top-color: var(--orange); }
[data-testid="stColumn"]:nth-of-type(4) [data-testid="stMetric"] { border-top-color: #2BB3A3; }
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.95rem; font-weight: 500; }
[data-testid="stMetricValue"] {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  color: var(--ink);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.4rem;
  background: #fff;
  padding: 0.4rem;
  border-radius: 16px;
  border: 1px solid var(--line);
  flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  font-weight: 600;
  color: var(--muted);
  height: auto;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(120deg, var(--violet), var(--magenta));
  color: #fff !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
.side-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.35rem;
  color: var(--ink);
  margin-bottom: 0.2rem;
}
.side-sub { color: var(--muted); font-size: 0.92rem; margin-bottom: 1rem; line-height: 1.45; }

/* Callouts and titles */
.callout {
  background: #fff;
  border: 1px solid var(--line);
  border-left: 5px solid var(--violet);
  border-radius: 14px;
  padding: 0.9rem 1.2rem;
  margin: 0.4rem 0 1.2rem 0;
  color: var(--ink);
  line-height: 1.55;
}
.callout strong { display: block; margin-bottom: 0.15rem; }
.callout span { color: var(--muted); }
.chart-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.15rem;
  color: var(--ink);
  margin: 0.6rem 0 0.4rem 0;
}
.test-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.15rem;
  margin: 0.8rem 0 0.2rem 0;
}

/* Winner badge and recommendation cards */
.badge {
  display: inline-block;
  background: linear-gradient(120deg, var(--violet), var(--magenta));
  color: #fff;
  border-radius: 999px;
  padding: 0.3rem 0.95rem;
  font-weight: 600;
  font-size: 0.95rem;
  margin-bottom: 0.6rem;
}
.rec-card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.2rem 1.3rem;
  box-shadow: 0 6px 20px rgba(60, 40, 140, 0.06);
  height: 100%;
}
.rec-card.first {
  border: 2px solid var(--magenta);
  box-shadow: 0 10px 28px rgba(214, 36, 110, 0.16);
}
.rec-rank { color: var(--muted); font-weight: 600; font-size: 0.9rem; }
.rec-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.35rem;
  line-height: 1.2;
  margin: 0.25rem 0 0.5rem 0;
  color: var(--ink);
}
.rec-meta { color: var(--muted); font-size: 0.95rem; line-height: 1.5; }
.rec-score {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800;
  font-size: 1.9rem;
  color: var(--magenta);
  margin-top: 0.6rem;
}
.rec-uplift { color: #1E8E5A; font-weight: 600; font-size: 0.95rem; }

/* Buttons */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(120deg, var(--violet), var(--magenta));
  color: #fff;
  border: none;
  border-radius: 12px;
  padding: 0.6rem 1.4rem;
  font-weight: 600;
  box-shadow: 0 6px 16px rgba(214, 36, 110, 0.25);
}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover { color: #fff; filter: brightness(1.08); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 3px solid var(--orange);
  outline-offset: 2px;
}

/* Forms, tables, alerts */
[data-testid="stForm"] {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.4rem;
  box-shadow: 0 6px 20px rgba(60, 40, 140, 0.06);
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: var(--line) !important; }

@media (max-width: 760px) {
  .hero { padding: 1.8rem 1.4rem; border-radius: 20px; }
  .hero h1 { font-size: 2rem; }
}
</style>
"""


def apply_theme():
    """Inject CSS and set the matplotlib/seaborn look. Call once, early."""
    st.markdown(CSS, unsafe_allow_html=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#EEEAF7",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
    })


def callout(title, body):
    st.markdown(
        f'<div class="callout"><strong>{title}</strong>'
        f'<span>{body}</span></div>',
        unsafe_allow_html=True,
    )


def chart_title(text):
    st.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)


def test_title(text):
    st.markdown(f'<div class="test-title">{text}</div>', unsafe_allow_html=True)


def show_df(data, **kwargs):
    """st.dataframe that works on old and new Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)