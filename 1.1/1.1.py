import streamlit as st
import numpy as np
import plotly.graph_objects as go
from math import sqrt

st.set_page_config(page_title="1.1 – Szeparábilis differenciálegyenletek", layout="wide")

# Helper functions
# TODO: Separate utility functions into a separate module

def slope_field_traces(f, x_range, y_range, density=11, seg_len=0.2):
    """Színezett iránymező – a vonalvastagság és szín a |m|-hez arányos."""
    xs = np.linspace(x_range[0], x_range[1], density)
    ys = np.linspace(y_range[0], y_range[1], density)
    mags, slopes = [], {}
    for x0 in xs:
        for y0 in ys:
            try:
                m = f(x0, y0)
            except Exception:
                continue
            if np.isfinite(m):
                slopes[(x0, y0)] = m
                mags.append(abs(m))
    if not mags:
        return []
    max_mag = max(mags)

    traces = []
    for (x0, y0), m in slopes.items():
        norm = (1 + m**2) ** 0.5
        dx, dy = seg_len / norm, m * seg_len / norm
        rel = abs(m) / max_mag
        color = f"rgb({int(255*rel)},0,{int(255*(1-rel))})"
        width = 1 + 2.5 * rel
        traces.append(
            go.Scatter(
                x=[x0 - dx, x0 + dx],
                y=[y0 - dy, y0 + dy],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def rk4(f, x0, y0, x_end, n_steps=1200):
    xs = np.linspace(x0, x_end, n_steps)
    ys = np.zeros_like(xs)
    ys[0] = y0
    for i in range(1, n_steps):
        h = xs[i] - xs[i - 1]
        k1 = f(xs[i - 1], ys[i - 1])
        k2 = f(xs[i - 1] + h / 2, ys[i - 1] + h * k1 / 2)
        k3 = f(xs[i - 1] + h / 2, ys[i - 1] + h * k2 / 2)
        k4 = f(xs[i - 1] + h, ys[i - 1] + h * k3)
        ys[i] = ys[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return xs, ys


def clip_curve(xs, ys, x_rng, y_rng):
    xmin, xmax = x_rng
    ymin, ymax = y_rng
    xs_out, ys_out = [], []
    for x, y in zip(xs, ys):
        if xmin <= x <= xmax and ymin <= y <= ymax:
            xs_out.append(x)
            ys_out.append(y)
        else:
            xs_out.append(None)
            ys_out.append(None)
    return xs_out, ys_out


# Plotly config
plot_layout = dict(width=600, height=600, hovermode="closest", margin=dict(l=0, r=0, t=10, b=10))


def axis_cfg(rng, dtick):
    """Main config style used for the plotly interactive plots."""
    return dict(
        range=rng,
        dtick=dtick,
        scaleanchor="y",
        scaleratio=1,
        constrain="range",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="gray",
        zerolinewidth=1,
        showline=True,
        mirror=True,
        linecolor="gray",
        linewidth=1,
    )


# Title
st.title("1.1 – Szeparábilis differenciálegyenletek")

st.header("Szeparábilis egyenletek megoldása")

# Ex. 1.  (dy/dx = -6xy , y(0)=7)
st.subheader("Példa 1")
col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.latex(r"\frac{dy}{dx} = -6xy,\qquad y(0)=7")
    st.markdown("**Megoldás lépései:**")
    st.latex(r"\frac{dy}{y} = -6x\,dx \;\Rightarrow\; \int \frac{dy}{y}=\int -6x\,dx")
    st.latex(r"\ln y = -3x^{2}+C \;\Rightarrow\; y = Ae^{-3x^{2}}")
    st.markdown("A kezdeti feltétel $y(0)=7$ miatt $A=7$, így")
    st.latex(r"y(x)=7e^{-3x^{2}}")
with col2:
    f_p1 = lambda x, y: -6*x*y
    rng_p1 = (-4, 4)
    figp1 = go.Figure()
    figp1.add_traces(slope_field_traces(f_p1, rng_p1, rng_p1))
    xs = np.linspace(rng_p1[0], rng_p1[1], 800)
    ys = 7*np.exp(-3*xs**2)
    xs, ys = clip_curve(xs, ys, rng_p1, rng_p1)
    figp1.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Megoldás", line=dict(width=4)))
    figp1.update_layout(xaxis=axis_cfg(rng_p1, 2), yaxis=axis_cfg(rng_p1, 2), **plot_layout)
    st.plotly_chart(figp1)

# Ex. 2.  (dy/dx = (4-2x)/(3y^2-5), y(1)=3)
st.subheader("Példa 2")
col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.latex(r"\frac{dy}{dx}=\frac{4-2x}{3y^{2}-5},\qquad y(1)=3")
    st.markdown("**Megoldás lépései:**")
    st.latex(r"(3y^{2}-5)\,dy = (4-2x)\,dx")
    st.latex(r"\int (3y^{2}-5)\,dy = \int (4-2x)\,dx")
    st.latex(r"y^{3}-5y = 4x - x^{2} + C")
    st.markdown("A kezdeti feltétel $(1,3)$ alapján $C=9$, így")
    st.latex(r"y^{3}-5y = 4x - x^{2} + 9")
with col2:
    f_p2 = lambda x, y: (4-2*x)/(3*y**2-5) if (3*y**2-5)!=0 else 0
    rng_p2 = (-2, 2)
    figp2 = go.Figure()
    figp2.add_traces(slope_field_traces(f_p2, rng_p2, rng_p2))
    xs_fwd, ys_fwd = rk4(f_p2, 1, 3, rng_p2[1])
    xs_bwd, ys_bwd = rk4(f_p2, 1, 3, rng_p2[0])
    xs = np.concatenate((xs_bwd[::-1], xs_fwd))
    ys = np.concatenate((ys_bwd[::-1], ys_fwd))
    xs, ys = clip_curve(xs, ys, rng_p2, rng_p2)
    figp2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Part. megoldás", line=dict(width=4)))
    figp2.update_layout(xaxis=axis_cfg(rng_p2,1), yaxis=axis_cfg(rng_p2,1), **plot_layout)
    st.plotly_chart(figp2)
    
st.header("Fontos alkalmazások")

# TODO

st.header("Ajánlott feladatok")

# TODO 

st.header("Házi feladatok")

# EP. EDE. 14.

st.subheader("1. Házi feladat [14]")
col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.latex(r"\frac{dy}{dx}=\frac{1+\sqrt{x}}{1+\sqrt{y}}")
    st.markdown("**Implicit megoldás:**")
    st.latex(r"y+\tfrac{2}{3}y^{3/2}=x+\tfrac{2}{3}x^{3/2}+C")

with col2:
    def f14(x, y):
        if x < 0 or y < 0:
            raise ValueError
        return (1 + sqrt(x)) / (1 + sqrt(y))

    rng14 = (0, 10)
    fig14 = go.Figure()
    fig14.add_traces(slope_field_traces(f14, rng14, rng14))

    for y0 in [0, 2, 4]:
        xs, ys = rk4(f14, 0, y0, rng14[1])
        xs, ys = clip_curve(xs, ys, rng14, rng14)
        fig14.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name=f"IV: (0,{y0})", line=dict(width=3))
        )

    fig14.update_layout(xaxis=axis_cfg(rng14, 2), yaxis=axis_cfg(rng14, 2), **plot_layout)
    st.plotly_chart(fig14)


# EP. EDE. 17.
st.subheader("2. házi feladat [17]")
col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.latex(r"y' = 1 + x + y + xy")
    st.markdown("**Megoldás:**")
    st.latex(r"y(x)=C\,e^{x+\tfrac{x^{2}}{2}}-1")

with col2:
    f17 = lambda x, y: (1 + x) * (1 + y)
    rng17 = (-10, 10)
    fig17 = go.Figure()
    fig17.add_traces(slope_field_traces(f17, rng17, rng17))

    for C in [-1, 0.001, 3]:
        xs = np.linspace(rng17[0], rng17[1], 2500)
        ys = C * np.exp(xs + xs**2 / 2) - 1
        xs, ys = clip_curve(xs, ys, rng17, rng17)
        fig17.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"C={C}", line=dict(width=3)))

    fig17.update_layout(xaxis=axis_cfg(rng17, 5), yaxis=axis_cfg(rng17, 5), **plot_layout)
    st.plotly_chart(fig17)


# EP. EDE. 28.
st.subheader("3. házi feladat [28]")
col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.latex(r"2\sqrt{x}\,\frac{dy}{dx}=\cos^{2}y,\qquad y(4)=\pi/4")
    st.markdown("**Megoldás:**")
    st.latex(r"y(x)=\arctan(\sqrt{x}-1)")

with col2:
    f28 = lambda x, y: (np.cos(y) ** 2) / (2 * sqrt(x)) if x > 0 else 0
    rng28 = (0, 10)
    fig28 = go.Figure()
    fig28.add_traces(slope_field_traces(f28, rng28, rng28))

    xs = np.linspace(0.01, rng28[1], 1800)
    ys_ivp = np.arctan(np.sqrt(xs) - 1)
    xs_ivp, ys_ivp = clip_curve(xs, ys_ivp, rng28, rng28)
    fig28.add_trace(go.Scatter(x=xs_ivp, y=ys_ivp, mode="lines", name="IVP", line=dict(width=4)))

    for C in [-2, 1]:
        ysC = np.arctan(np.sqrt(xs) + C)
        xsC, ysC = clip_curve(xs, ysC, rng28, rng28)
        fig28.add_trace(
            go.Scatter(x=xsC, y=ysC, mode="lines", name=f"C'={C}", line=dict(width=3, dash="dot"))
        )

    fig28.update_layout(xaxis=axis_cfg(rng28, 2), yaxis=axis_cfg(rng28, 2), **plot_layout)
    st.plotly_chart(fig28)


# EP. EDE. 48.
st.subheader("Az univerzum korának becslése [48]")
col1, col2 = st.columns([1, 2], gap="medium")
with col1:
    st.markdown("**Radioaktív bomlás modell:**")
    st.latex(r"N_i(t)=N_{0,i}e^{-\lambda_i t}\quad (i=235,238)")
    st.latex(r"\lambda_i=\frac{\ln 2}{T_{1/2,i}}")
    st.markdown("Félidők:  ")
    st.markdown("- $T_{1/2}^{238}=4.51\times10^9$ év  ")
    st.markdown("- $T_{1/2}^{235}=7.10\times10^8$ év")

with col2:
    half_238 = 4.51e9
    half_235 = 7.10e8
    lam_238 = np.log(2) / half_238
    lam_235 = np.log(2) / half_235

    t = np.linspace(0, 1e10, 1500)  # 0–10 Gyr
    N238 = np.exp(-lam_238 * t)
    N235 = np.exp(-lam_235 * t)

    figU = go.Figure()
    figU.add_trace(go.Scatter(x=t / 1e9, y=np.log10(N238), mode="lines", name="log₁₀ N₂₃₈U", line=dict(width=3)))
    figU.add_trace(go.Scatter(x=t / 1e9, y=np.log10(N235), mode="lines", name="log₁₀ N₂₃₅U", line=dict(width=3)))

    figU.update_layout(xaxis_title="t [Gyr]", yaxis_title="log₁₀ N/N₀", **plot_layout)
    st.plotly_chart(figU)

st.markdown("---")
st.markdown("Készítette: *Monori János Bence* - *Differenciálegyenletek - interaktív oktatási segédanyagok*.")
