"""
app_streamlit.py — Tablero de cartera de autos · Módulo 5 · Tema 1
==================================================================
Diplomado Machine Learning en Seguros · FC UNAM

Tablero con filtros, KPIs de riesgo y rentabilidad, gráficas por factor y un
cotizador con tres primas en orden: 1) empírica → 2) GLM → 3) prima de tarifa.

Correr local:
    pip install -r requirements.txt
    streamlit run app_streamlit.py

Versión preparada para deploy en Streamlit Community Cloud:
  · Rutas ancladas a la carpeta del archivo (AQUI), iguales en laptop y servidor.
  · Lee datos/datos.parquet y reconstruye cartera.db sola la primera vez.
"""

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import gspread

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Configuración de la página y rutas
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Tablero de cartera — Diplomado ML Seguros",
                   page_icon="📊", layout="wide")

# Rutas ancladas a la carpeta de este archivo (funcionan igual local y en el servidor)
AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_PARQUET = os.path.join(AQUI, "datos", "datos.parquet")
RUTA_DB = os.path.join(AQUI, "cartera.db")

# Bandas de M4 (mismos cortes que en los GLM del Módulo 4)
BANDAS_EDAD = [17, 30, 35, 45, 50, 55, 60, 95]
BANDINGS = {"edad_conductor": [17, 30, 35, 45, 50, 55, 60, 95],
            "antiguedad_vehiculo": [-1, 1, 2, 3, 4, 5, 10, 15, 50],
            "potencia": [9, 40, 50, 60, 70, 250],
            "nivel_bonus": [-1, 0, 1, 2, 5, 10, 25]}


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Carga de datos (cacheada): parquet → SQLite → DataFrame
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    if not os.path.exists(RUTA_PARQUET):
        st.error("No encuentro datos/datos.parquet. Revisa que esté en la carpeta datos/ del repo.")
        st.stop()

    # La base no se sube al repo: se reconstruye desde el parquet la primera vez
    if not os.path.exists(RUTA_DB):
        con0 = sqlite3.connect(RUTA_DB)
        pd.read_parquet(RUTA_PARQUET).to_sql("cartera", con0, if_exists="replace", index=False)
        con0.close()

    con = sqlite3.connect(RUTA_DB)
    df = pd.read_sql("SELECT * FROM cartera", con)
    con.close()

    df["edad_cat"] = pd.cut(df["edad_conductor"], bins=BANDAS_EDAD)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2 · KPIs y modelos
# ─────────────────────────────────────────────────────────────────────────────
def kpi_exposicion(df):      return df["exposicion"].sum()
def kpi_num_siniestros(df):  return int(df["num_siniestros"].sum())
def kpi_frecuencia(df):      return df["num_siniestros"].sum() / df["exposicion"].sum()


def kpi_severidad_media(df):
    """Ponderada por nº de siniestros, solo sobre pólizas con N>0."""
    con = df[df["num_siniestros"] > 0]
    if con["num_siniestros"].sum() == 0:
        return np.nan
    return np.average(con["severidad"], weights=con["num_siniestros"])


def kpi_prima_pura(df):      return df["monto_total"].sum() / df["exposicion"].sum()


def kpis_por(df, variable):
    """Tabla de KPIs por nivel de una variable, ordenada por exposición."""
    filas = []
    for nivel, g in df.groupby(variable, observed=True):
        filas.append({
            variable: str(nivel),
            "Exposición": kpi_exposicion(g),
            "Nº sin.": kpi_num_siniestros(g),
            "Frecuencia": kpi_frecuencia(g),
            "Severidad": kpi_severidad_media(g),
            "Prima pura": kpi_prima_pura(g),
        })
    return pd.DataFrame(filas).sort_values("Exposición", ascending=False).reset_index(drop=True)


def banda(col, val):
    return str(pd.cut([val], bins=BANDINGS[col])[0])


def cotizar_empirico(d, p):
    """Prima pura empírica del segmento. Se segmenta solo por variables con suficiente
    volumen (edad en banda, cobertura, sexo, uso, combustible) para no quedar vacío."""
    seg = d[(d["edad_conductor_cat"] == banda("edad_conductor", p["edad_conductor"])) &
            (d["cobertura"] == p["cobertura"]) &
            (d["sexo"] == p["sexo"]) &
            (d["uso"] == p["uso"]) &
            (d["combustible"] == p["combustible"])]
    n = len(seg)
    expo = seg["exposicion"].sum()
    pp = seg["monto_total"].sum() / expo if expo > 0 else np.nan
    return pp, n


@st.cache_resource
def ajustar_modelos(d):
    d = d.copy()
    sev = d[d["num_siniestros"] > 0].copy()
    m_freq = smf.glm(
        "num_siniestros ~ C(nivel_bonus_cat)+C(edad_conductor_cat)+C(combustible)+"
        "C(antiguedad_vehiculo_cat)+C(potencia_cat)+C(cobertura)+C(uso)",
        d, family=sm.families.Poisson(), offset=np.log(d["exposicion"])).fit()
    m_sev = smf.glm(
        "severidad ~ C(edad_conductor_cat)+C(antiguedad_vehiculo_cat)+C(potencia_cat)+"
        "C(nivel_bonus_cat)+C(sexo)",
        sev, family=sm.families.Gamma(sm.families.links.Log()),
        freq_weights=sev["num_siniestros"]).fit()

    d["pure_premium"] = d["monto_total"] / d["exposicion"]
    m_agg = smf.glm(
        "pure_premium ~ C(cobertura)+C(sexo)+C(combustible)+C(edad_conductor_cat)+"
        "C(potencia_cat)+C(nivel_bonus_cat)",
        d, family=sm.families.Tweedie(var_power=1.3, link=sm.families.links.Log()),
        var_weights=d["exposicion"]).fit()
    return m_freq, m_sev, m_agg


def cotizar_glm(p, m_freq, m_sev, m_agg):
    row = pd.DataFrame([{
        "edad_conductor_cat": banda("edad_conductor", p["edad_conductor"]),
        "antiguedad_vehiculo_cat": banda("antiguedad_vehiculo", p["antiguedad"]),
        "potencia_cat": banda("potencia", p["potencia"]),
        "nivel_bonus_cat": banda("nivel_bonus", p["bonus"]),
        "cobertura": p["cobertura"],
        "sexo": p["sexo"],
        "uso": p["uso"],
        "combustible": p["combustible"],
        "exposicion": 1.0}])
    freq = float(m_freq.predict(row, offset=np.log(row["exposicion"]))[0])
    sev = float(m_sev.predict(row)[0])
    pp_tw = float(m_agg.predict(row)[0])
    return freq, sev, freq * sev, pp_tw


NOMBRE = {"edad_conductor_cat": "edad", "nivel_bonus_cat": "nivel de bonus", "potencia_cat": "potencia",
          "antiguedad_vehiculo_cat": "antigüedad del vehículo", "cobertura": "cobertura",
          "sexo": "sexo", "combustible": "combustible", "uso": "uso"}


def relatividades(p, m_freq, m_sev):
    def rel(m, pref, niv):                      # factor del nivel (1.0 si es la referencia)
        return float(np.exp(m.params.get(f"C({pref})[T.{niv}]", 0.0)))
    nivs = {"edad_conductor_cat": banda("edad_conductor", p["edad_conductor"]),
            "nivel_bonus_cat":    banda("nivel_bonus", p["bonus"]),
            "potencia_cat":       banda("potencia", p["potencia"]),
            "antiguedad_vehiculo_cat": banda("antiguedad_vehiculo", p["antiguedad"]),
            "cobertura": p["cobertura"], "sexo": p["sexo"],
            "combustible": p["combustible"], "uso": p["uso"]}
    out = [(NOMBRE[pref], str(niv), rel(m_freq, pref, niv) * rel(m_sev, pref, niv))
           for pref, niv in nivs.items()]
    return sorted(out, key=lambda x: abs(np.log(x[2])), reverse=True)   # mayor impacto primero


# ─────────────────────────────────────────────────────────────────────────────
# 2b · Memoria: guardar cada cotización en Google Sheets (bonus)
#      Credenciales vía st.secrets (local: .streamlit/secrets.toml, que NO se sube;
#      servidor: App settings → Secrets). Si no hay secrets, la app sigue funcionando.
# ─────────────────────────────────────────────────────────────────────────────
def sheets_configurado():
    try:
        return "gcp_service_account" in st.secrets and "sheet_url" in st.secrets
    except Exception:
        return False


@st.cache_resource
def hoja():
    gc = gspread.service_account_from_dict(dict(st.secrets["gcp_service_account"]))
    return gc.open_by_url(st.secrets["sheet_url"]).sheet1


def ahora_texto():
    try:
        return pd.Timestamp.now(tz="America/Mexico_City").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def guardar_cotizacion(p, pp_glm, tarifa):
    if not sheets_configurado():
        st.info("La cotización no se guardó: falta configurar Google Sheets en los secrets.")
        return
    try:
        hoja().append_row([ahora_texto(),
                           int(p["edad_conductor"]), str(p["cobertura"]), str(p["sexo"]),
                           str(p["uso"]), str(p["combustible"]), int(p["antiguedad"]),
                           int(p["potencia"]), int(p["bonus"]),
                           round(float(pp_glm), 2), round(float(tarifa), 2)],
                          value_input_option="USER_ENTERED")
        st.success("Cotización guardada en Google Sheets ✔")
    except Exception as e:
        st.error(f"No se pudo guardar la cotización en Google Sheets: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Cuerpo de la app
# ─────────────────────────────────────────────────────────────────────────────
datos = cargar_datos()

for col, cortes in BANDINGS.items():
    datos[col + "_cat"] = pd.cut(datos[col], bins=cortes).astype(str)
m_freq, m_sev, m_agg = ajustar_modelos(datos)


@st.cache_data
def agregar_prima_glm(datos, _m_freq, _m_sev):
    out = datos.copy()
    out["prima_pura_glm"] = (_m_freq.predict(out, offset=np.log(out["exposicion"])) * (_m_sev.predict(out)))
    return out


datos = agregar_prima_glm(datos, m_freq, m_sev)


def ratios(sub, gastos, margen):
    primas = (sub["prima_pura_glm"] / (1 - gastos - margen)).sum()
    if primas == 0:
        return np.nan, np.nan
    loss = sub["monto_total"].sum() / primas
    return loss, loss + gastos


st.title("📊 Tablero de cartera — auto")
st.caption("Cartera del Módulo 4 · mueve los filtros de la izquierda y prueba el cotizador.")

# --- Barra lateral: filtros ---------------------------------------------------
st.sidebar.header("Filtros")

with st.sidebar.form("filtros"):
    f_cob = st.multiselect("Cobertura", options=sorted(datos["cobertura"].unique()),
                           default=sorted(datos["cobertura"].unique()),
                           help="Tipo de cobertura de la póliza")
    f_sexo = st.multiselect("Sexo", options=sorted(datos["sexo"].unique()),
                            default=sorted(datos["sexo"].unique()),
                            help="Sexo del contratante de la póliza")
    f_uso = st.multiselect("Uso", options=sorted(datos["uso"].unique()),
                           default=sorted(datos["uso"].unique()),
                           help="Uso del vehículo asegurado")
    f_comb = st.multiselect("Combustible", options=sorted(datos["combustible"].unique()),
                            default=sorted(datos["combustible"].unique()),
                            help="Tipo de combustible del vehículo asegurado")
    emin, emax = int(datos["edad_conductor"].min()), int(datos["edad_conductor"].max())
    f_edad = st.slider("Edad", min_value=emin, max_value=emax, value=(emin, emax),
                       help="Rango de edad del conductor de la póliza")
    st.form_submit_button("Aplicar filtros")

df = datos[datos["cobertura"].isin(f_cob) & datos["sexo"].isin(f_sexo) &
           datos["uso"].isin(f_uso) & datos["combustible"].isin(f_comb)]
df = df[df["edad_conductor"].between(*f_edad)]

if len(df) == 0:
    st.warning("Ningún registro con esos filtros. Amplía la selección.")
    st.stop()

st.sidebar.header("Recargos de tarifa")
gastos = st.sidebar.slider("Gastos", 0.0, 0.5, 0.25, 0.01)
margen = st.sidebar.slider("Margen", 0.0, 0.3, 0.05, 0.01)

tab_cartera, tab_detalle, tab_cotiza = st.tabs(["Cartera", "Detalle", "Cotizador"])

with tab_cartera:
    df = df.copy()
    st.sidebar.metric("Pólizas seleccionadas", f"{len(df):,}")

    # --- Tarjetas de KPI ------------------------------------------------------
    st.subheader("Indicadores de riesgo")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Exposición", f"{kpi_exposicion(df):,.0f}")
    c2.metric("Nº siniestros", f"{kpi_num_siniestros(df):,}")
    c3.metric("Frecuencia", f"{kpi_frecuencia(df):.4f}",
              delta=f"{kpi_frecuencia(df) - kpi_frecuencia(datos):+.4f} vs. cartera completa")
    c4.metric("Severidad media", f"${kpi_severidad_media(df):,.0f}",
              delta=f"${kpi_severidad_media(df) - kpi_severidad_media(datos):,.0f}")
    c5.metric("Prima pura", f"${kpi_prima_pura(df):,.2f}")

    st.subheader("Indicadores de rentabilidad")
    loss, comb_ratio = ratios(df, gastos, margen)
    r1, r2 = st.columns(2)
    r1.metric("Loss ratio", f"{loss:.1%}")
    r2.metric("Combined ratio", f"{comb_ratio:.1%}",
              delta=("rentable" if comb_ratio < 1 else "pérdida"),
              delta_color=("normal" if comb_ratio < 1 else "inverse"))

    st.divider()

    # --- Gráficas -------------------------------------------------------------
    g1, g2 = st.columns(2)

    with g1:
        with st.container(border=True):
            st.markdown("**Exposición por banda de edad**")
            tab_edad = kpis_por(df, "edad_cat")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            ax1.bar(tab_edad["edad_cat"], tab_edad["Exposición"], color="#4C72B0")
            ax1.set_xlabel("Edad"); ax1.set_ylabel("Años-póliza")
            ax1.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right"); fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

    with g2:
        with st.container(border=True):
            factor = st.selectbox("Factor", ["cobertura", "sexo", "uso", "combustible"], index=0)
            st.markdown(f"**Combined ratio por {factor}**")
            rows = [(str(niv), ratios(g, gastos, margen)[1])
                    for niv, g in df.groupby(factor, observed=True)]
            fig2, ax2 = plt.subplots(figsize=(6, 3.2))
            ax2.bar([r[0] for r in rows], [r[1] for r in rows])
            ax2.axhline(1.0, ls="--", color="red")
            ax2.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=45, ha="right"); fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    st.divider()

with tab_detalle:
    st.subheader("KPIs por segmento")
    var_tabla = st.selectbox("Segmentar por", ["cobertura", "sexo", "uso", "combustible", "edad_cat"], index=0)
    tabla = kpis_por(df, var_tabla)
    st.dataframe(
        tabla.style.format({
            "Exposición": "{:,.0f}", "Nº sin.": "{:,}",
            "Frecuencia": "{:.4f}", "Severidad": "${:,.0f}", "Prima pura": "${:,.2f}",
        }),
        hide_index=True,
    )

    csv = tabla.to_csv(index=False).encode("utf-8")
    st.download_button(label="Descargar tabla como CSV", data=csv,
                       file_name=f"tabla_kpis_por_{var_tabla}.csv", mime="text/csv")

    with st.expander("¿Qué significan estas columnas?"):
        st.write("Frecuencia = Σ siniestros / Σ exposición · Severidad ponderada por "
                 "nº de siniestros · Prima pura = Σ monto / Σ exposición.")

with tab_cotiza:
    st.subheader("Cotizador de prima")
    st.caption("Tres primas en orden: 1) empírica del segmento → 2) GLM → 3) prima de tarifa.")

    c1, c2, c3, c4 = st.columns(4)

    perfil = {
        "edad_conductor": c1.slider("Edad", 18, 95, 35),
        "cobertura": c1.selectbox("Cobertura", options=sorted(datos["cobertura"].unique()), index=0),
        "sexo": c2.selectbox("Sexo", options=sorted(datos["sexo"].unique()), index=0),
        "uso": c2.selectbox("Uso", options=sorted(datos["uso"].unique()), index=0),
        "combustible": c3.selectbox("Combustible", options=sorted(datos["combustible"].unique()), index=0),
        "antiguedad": c3.slider("Antigüedad del vehículo", 0, 40, 5),
        "potencia": c4.slider("Potencia del vehículo (HP)", 10, 200, 70),
        "bonus": c4.slider("Bonus-malus", 0, 25, 5),
    }

    if st.button("Cotizar", type="primary"):
        st.session_state["perfil"] = perfil
        # Calcula la prima GLM y la tarifa de este perfil y la guarda como fila nueva
        _, _, pp_glm_nuevo, _ = cotizar_glm(perfil, m_freq, m_sev, m_agg)
        guardar_cotizacion(perfil, pp_glm_nuevo, pp_glm_nuevo / (1 - gastos - margen))

    if "perfil" in st.session_state:
        p = st.session_state["perfil"]
        st.info("Perfil cotizado: " + str(p))

        # 1 · Empírica
        pp_emp, n = cotizar_empirico(datos, p)
        with st.container(border=True):
            st.markdown("**1 · Cotización empírica**")
            if n < 100:
                st.warning(f"Solo {n} pólizas en este segmento. La prima empírica puede ser poco creíble.")
            st.metric("Prima pura empírica", "-" if np.isnan(pp_emp) else f"${pp_emp:,.2f}",
                      delta=f"{n} pólizas en el segmento", delta_color="off")

        # 2 · GLM
        freq, sev, pp_glm, pp_tw = cotizar_glm(p, m_freq, m_sev, m_agg)
        with st.container(border=True):
            st.markdown("**2 · Cotización por GLM**")
            st.metric("Prima pura (freq × sev)", f"${pp_glm:,.2f}")
            st.caption(f"Frecuencia predicha: {freq:.4f} · Severidad predicha: ${sev:,.0f}")
            st.metric("Prima pura (Tweedie)", f"${pp_tw:,.2f}")

        # 3 · Tarifa
        tarifa = pp_glm / (1 - gastos - margen)
        with st.container(border=True):
            st.markdown("**3 · Precio**")
            st.metric("Prima de TARIFA", f"${tarifa:,.2f}")
            st.caption(f"Prima pura GLM / (1 − gastos {gastos:.0%} − margen {margen:.0%})")

        st.subheader("¿Por qué esta prima?")
        rels = relatividades(p, m_freq, m_sev)
        pp_ref = pp_glm / np.prod([r for _, _, r in rels])      # prima del perfil de referencia

        etiquetas = [f"{nombre}: {niv}" for nombre, niv, r in rels]
        pct = [(r - 1) * 100 for _, _, r in rels]
        colores = ["#C44E52" if v > 0 else "#2A9D8F" for v in pct]
        fig3, ax3 = plt.subplots(figsize=(8, 0.5 * len(rels) + 0.6))
        ax3.barh(etiquetas, pct, color=colores)
        ax3.axvline(0, color="#555", lw=1)
        ax3.set_xlabel("Ajuste sobre la prima base (%)")
        ax3.spines[["top", "right"]].set_visible(False)
        for y, v in enumerate(pct):
            ax3.text(v + (1 if v >= 0 else -1), y, f"{v:+.0f}%", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9)
        ax3.invert_yaxis()
        ax3.margins(x=0.18)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
        st.caption(f"Prima base (perfil de referencia): ${pp_ref:,.2f} → tu prima pura: ${pp_glm:,.2f}. "
                   "Rojo = pagas más · verde = pagas menos.")

    if sheets_configurado():
        with st.expander("Últimas cotizaciones guardadas (Google Sheets)"):
            try:
                registros = pd.DataFrame(hoja().get_all_records())
                if registros.empty:
                    st.write("Todavía no hay cotizaciones guardadas.")
                else:
                    st.dataframe(registros.tail(10), hide_index=True)
            except Exception as e:
                st.error(f"No se pudo leer la Google Sheet: {e}")
