print("Librerías cargadas.")
import pandas as pd  
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import seaborn as sns
import unicodedata
from fuzzywuzzy import process
import re
from adjustText import adjust_text
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from itertools import combinations

# Carpetas de directorio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(BASE_DIR)

def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s)
    return s
# Coordenadas directas (42 sucursales)
# === Helpers de texto para conclusiones (no requieren imports nuevos) ===
_MONEY_COLS__RIPLEY = {'Total_Notas_Credito','Costo_Merma_Destruccion','Productos_Dañados_Descuento','Total_Ventas','Ventas / Mts. 2'}
_PCT_COLS__RIPLEY   = {'Desviacion_Meta_Merma','Rot_2025','NC / Ventas'}
_COUNT_COLS__RIPLEY = {'Reclamos_Cantidad_2025','Juicios_Laborales','N° Accidentes','N° Multas','N° Recuperos 2025'}

def _fmt_num_es_ripley(v, dec=0, signed=False):
    import math, pandas as pd
    if pd.isna(v): return "s/d"
    sgn = "-" if signed and float(v) < 0 else ""
    v = abs(float(v))
    txt = f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return sgn + txt
 
def _fmt_money_clp_ripley(v):
    return f"${_fmt_num_es_ripley(v, dec=0)}" if v is not None else "s/d"
 
def _fmt_pct__ripley(v, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "s/d"
    try:
        return f"{float(v)*100:.{dec}f}%"
    except Exception:
        return str(v)

def _fmt_val_ripley(col, v):
    if col in _PCT_COLS__RIPLEY:   return _fmt_pct__ripley(v, dec=1)
    if col in _MONEY_COLS__RIPLEY: return _fmt_money_clp_ripley(v)
    if col in _COUNT_COLS__RIPLEY: return _fmt_num_es_ripley(v, dec=0)
    if col == 'Indice_Inseguridad': return _fmt_num_es_ripley(v, dec=2)
    return _fmt_num_es_ripley(v, dec=2)


_ALIAS_TXT_RIPLEY = {
    'Total_Notas_Credito': 'monto en Notas de crédito',
    'Reclamos_Cantidad_2025': 'número de reclamos',
    'Juicios_Laborales': 'juicios laborales',
    'N° Accidentes': 'accidentes de colaboradores',
    'N° Multas': 'multas',
    'N° Recuperos 2025': 'recuperos de artículos',
    'Desviacion_Meta_Merma': 'desviación de la meta de merma',
    'Costo_Merma_Destruccion': 'merma por destrucción',
    'Productos_Dañados_Descuento': 'descuentos por productos dañados',
    'Indice_Inseguridad': 'índice de inseguridad',
    'Rot_2025': 'rotación de personal',
    'Ventas / Mts. 2': 'ventas por m²',
    'Total_Ventas': 'ventas totales',
}
 # === Helper final para construir 'Conclusiones' (REEMPLAZA la versión anterior) ===
def construir_conclusiones_ripley(reporte_base, X_minmax, pesos_dict, reporte_final, top_n=3):
    """
    Reglas actualizadas:
      - Excluir 'Ventas / Mts. 2' e 'Indice_Inseguridad'.
      - 'Desviacion_Meta_Merma' solo si es NEGATIVA; mostrar con 6 decimales (no %).
      - 'NC / Ventas': incluir p95 en el texto; añadir si ≥ p95 o si lo disparó la regla.
      - 'Costo_Merma_Destruccion':
            * usar MAGNITUD ABSOLUTA (|valor|) para decidir si aparece (distancia a 0),
            * por contribución: exigir |valor| >= p60(|x|),
            * por outlier: promover si |valor| >= p95(|x|),
            * si |valor| es el MÁXIMO absoluto del conjunto, forzar su inclusión al inicio.
      - 'Productos_Dañados_Descuento':
            * por contribución: exigir valor >= p60,
            * por outlier: promover si ≥ p95,
            * forzar inclusión si Motivo_Regla lo indica o si valor ≥ p85.
      - 'N° Recuperos 2025' y 'Reclamos_Cantidad_2025': promover desde p90 (y si es máximo).
      - Riesgo Alto: 4 drivers; Riesgo Medio: 3 drivers.
    """

    _EXCLUIR = {'Ventas / Mts. 2', 'Indice_Inseguridad'}
    _SOLO_NEGATIVO = {'Desviacion_Meta_Merma', 'Desviacion Meta Merme'}
    _PRIORIDAD_OUTLIERS = [
        'N° Accidentes','Total_Notas_Credito','Reclamos_Cantidad_2025','N° Recuperos 2025',
        'Juicios_Laborales','N° Multas','Costo_Merma_Destruccion','Productos_Dañados_Descuento',
        'Rot_2025','Desviacion_Meta_Merma','Desviacion Meta Merme',
    ]
    _UMBRAL_ESPECIFICO_Q = {'N° Recuperos 2025': 0.90, 'Reclamos_Cantidad_2025': 0.90}  # resto p95
    _FILTRAR_ALTO_CONTRIB_Q = {'Productos_Dañados_Descuento': 0.60}  # exige valor >= p60 por contribución
    _FORZAR_SI_GE_Q = {'Productos_Dañados_Descuento': 0.85}          # fuerza si ≥ p85
    _ABS_MAG_COLS = {'Costo_Merma_Destruccion'}                       # evaluar por |valor|

    # Contribuciones coherentes con tu score
    w = pd.Series(pesos_dict).reindex(X_minmax.columns).fillna(0.0)
    contrib = X_minmax.mul(w, axis=1)

    # Estadísticos (incluye absolutos para merma destrucción)
    def _stats(s):
        s = pd.to_numeric(s, errors='coerce')
        if not s.notna().any():
            return {'max': np.nan,'q95': np.nan,'q90': np.nan,'q85': np.nan,'q60': np.nan,'q50': np.nan,'q05': np.nan,
                    'abs_max': np.nan,'abs_q95': np.nan,'abs_q90': np.nan,'abs_q60': np.nan}
        s_abs = s.abs()
        return {
            'max': s.max(skipna=True),'q95': s.quantile(0.95),'q90': s.quantile(0.90),
            'q85': s.quantile(0.85),'q60': s.quantile(0.60),'q50': s.quantile(0.50),'q05': s.quantile(0.05),
            'abs_max': s_abs.max(skipna=True),'abs_q95': s_abs.quantile(0.95),
            'abs_q90': s_abs.quantile(0.90),'abs_q60': s_abs.quantile(0.60),
        }

    stats = {}
    cols_stats = set(_PRIORIDAD_OUTLIERS) | set(_FILTRAR_ALTO_CONTRIB_Q) | set(_FORZAR_SI_GE_Q) | _ABS_MAG_COLS
    for col in cols_stats:
        if col in reporte_base.columns:
            stats[col] = _stats(reporte_base[col])

    # p95 para NC/Ventas
    p95_ncv = np.nan
    if 'NC / Ventas' in reporte_final.columns:
        ncv_series = pd.to_numeric(reporte_final['NC / Ventas'], errors='coerce')
        if ncv_series.notna().any():
            p95_ncv = ncv_series.quantile(0.95)

    def _es_negativo(col, v):
        try:
            return (col in _SOLO_NEGATIVO) and (float(pd.to_numeric(v, errors='coerce')) < 0)
        except Exception:
            return False

    def _valor_tal_cual(col, raw_val):
        # Desviación: 6 decimales, sin %
        if col in _SOLO_NEGATIVO:
            try:
                fv = float(pd.to_numeric(raw_val, errors='coerce'))
                return f"{fv:.6f}"
            except Exception:
                return str(raw_val)
        return _fmt_val_ripley(col, raw_val)

    def _detallar_outlier(col, v):
        if col == 'NC / Ventas':
            if v is None or pd.isna(v): return ""
            tag_p95 = f" (p95={_fmt_pct__ripley(p95_ncv)})" if pd.notna(p95_ncv) else ""
            try:
                ge = (pd.notna(p95_ncv) and float(v) >= float(p95_ncv))
            except Exception:
                ge = False
            return (" (≥p95)" if ge else "") + tag_p95

        if col not in stats or v is None or pd.isna(v): return ""
        st = stats[col]
        try:
            fv = float(pd.to_numeric(v, errors='coerce'))
        except Exception:
            return ""

        if col in _SOLO_NEGATIVO:
            if fv < 0 and pd.notna(st['q05']) and fv <= st['q05']:
                return " (≤p5 del conjunto)"
            return ""

        if col in _ABS_MAG_COLS:
            fav = abs(fv)
            if pd.notna(st['abs_max']) and fav == st['abs_max']: return " (mayor |valor| del conjunto)"
            if pd.notna(st['abs_q95']) and fav >= st['abs_q95']: return " (|valor| ≥ p95 del conjunto)"
            return ""

        q_key = 'q95' if col not in _UMBRAL_ESPECIFICO_Q else 'q90'
        qv = st.get(q_key, np.nan)
        if pd.notna(st['max']) and fv == st['max']: return " (mayor del conjunto)"
        if pd.notna(qv) and fv >= qv: return f" (≥p{int(float(q_key[1:]))} del conjunto)"
        return ""

    textos = []
    for i in reporte_base.index:
        riesgo = reporte_final.loc[i, 'Riesgo_final'] if 'Riesgo_final' in reporte_final.columns else reporte_base.loc[i, 'Cluster_Riesgo']
        top_n_row = 4 if str(riesgo) == 'Riesgo Alto' else 3
        if riesgo not in ('Riesgo Medio', 'Riesgo Alto'):
            textos.append("—"); continue

        # Ranking por contribución (con filtros alto/abs y excluidos)
        c_sorted = contrib.loc[i].sort_values(ascending=False)
        cand_contrib = []
        for col in c_sorted.index:
            if w.get(col, 0) <= 0 or col in _EXCLUIR: continue
            if col in _SOLO_NEGATIVO:
                val = reporte_base.loc[i, col] if col in reporte_base.columns else None
                if not _es_negativo(col, val): continue
            if col in _FILTRAR_ALTO_CONTRIB_Q and col in reporte_base.columns:
                st = stats.get(col, {})
                vraw = pd.to_numeric(reporte_base.loc[i, col], errors='coerce')
                if pd.isna(vraw) or pd.isna(st.get('q60')) or float(vraw) < float(st['q60']): continue
            if col in _ABS_MAG_COLS and col in reporte_base.columns:
                st = stats.get(col, {})
                vraw = pd.to_numeric(reporte_base.loc[i, col], errors='coerce')
                if pd.isna(vraw) or pd.isna(st.get('abs_q60')) or abs(float(vraw)) < float(st['abs_q60']): continue
            cand_contrib.append(col)

        # Promoción de outliers
        cand_out = []
        for col in _PRIORIDAD_OUTLIERS:
            if col not in reporte_base.columns or col in _EXCLUIR: continue
            val = pd.to_numeric(reporte_base.loc[i, col], errors='coerce')
            if pd.isna(val): continue

            if col in _SOLO_NEGATIVO:
                if _es_negativo(col, val): cand_out.append(col)
            elif col in _ABS_MAG_COLS:
                st = stats.get(col, {})
                fav = abs(float(val))
                if (pd.notna(st.get('abs_max')) and fav == st['abs_max']) or (pd.notna(st.get('abs_q95')) and fav >= st['abs_q95']):
                    cand_out.append(col)
            else:
                st = stats.get(col, {})
                q_key = 'q95' if col not in _UMBRAL_ESPECIFICO_Q else 'q90'
                qv = st.get(q_key, np.nan)
                if (pd.notna(st.get('max')) and float(val) == st['max']) or (pd.notna(qv) and float(val) >= qv):
                    cand_out.append(col)

        # NC/Ventas driver si ≥ p95 o por regla
        incluir_ncv = False
        if 'NC / Ventas' in reporte_final.columns:
            ncv_val = pd.to_numeric(reporte_final.loc[i, 'NC / Ventas'], errors='coerce')
            if pd.notna(p95_ncv) and pd.notna(ncv_val) and float(ncv_val) >= float(p95_ncv):
                incluir_ncv = True

        # Forzar Dañados por regla o valor ≥ p85
        mot = str(reporte_final.loc[i, 'Motivo_Regla']) if 'Motivo_Regla' in reporte_final.columns and pd.notna(reporte_final.loc[i, 'Motivo_Regla']) else ""
        forzar_dan = False
        if 'Productos_Dañados_Descuento' in reporte_base.columns:
            st_dan = stats.get('Productos_Dañados_Descuento', {})
            val_dan = pd.to_numeric(reporte_base.loc[i, 'Productos_Dañados_Descuento'], errors='coerce')
            thr85 = st_dan.get('q85', np.nan)
            if ('Dañados' in mot) or (pd.notna(val_dan) and pd.notna(thr85) and float(val_dan) >= float(thr85)):
                forzar_dan = True

        # Unión: primero outliers, luego contribución
        seen, union = set(), []
        for col in (cand_out + cand_contrib):
            if col not in seen: union.append(col); seen.add(col)

        # Inserciones forzadas
        if incluir_ncv and 'NC / Ventas' not in union: union.insert(0, 'NC / Ventas')
        if forzar_dan:
            if 'Productos_Dañados_Descuento' in union: union.remove('Productos_Dañados_Descuento')
            union.insert(0, 'Productos_Dañados_Descuento')

        # Si es el mayor |valor| en Merma de destrucción, forzar al inicio
        if 'Costo_Merma_Destruccion' in reporte_base.columns and 'Costo_Merma_Destruccion' in stats:
            v_merma = pd.to_numeric(reporte_base.loc[i, 'Costo_Merma_Destruccion'], errors='coerce')
            st_merma = stats['Costo_Merma_Destruccion']
            if pd.notna(v_merma) and pd.notna(st_merma.get('abs_max')) and abs(float(v_merma)) == float(st_merma['abs_max']):
                if 'Costo_Merma_Destruccion' in union: union.remove('Costo_Merma_Destruccion')
                union.insert(0, 'Costo_Merma_Destruccion')

        top = union[:top_n_row]

        # Frases
        frases = []
        for col in top:
            if col == 'NC / Ventas':
                v = reporte_final.loc[i, 'NC / Ventas'] if 'NC / Ventas' in reporte_final.columns else np.nan
                detalle = _detallar_outlier('NC / Ventas', v)
                frases.append(f"NC/Ventas {_fmt_pct__ripley(v)}{detalle}")
                continue
            alias = _ALIAS_TXT_RIPLEY.get(col, col)
            val_raw = reporte_base.loc[i, col] if col in reporte_base.columns else None
            detalle = _detallar_outlier(col, val_raw)
            frases.append(f"{alias} {_valor_tal_cual(col, val_raw)}{detalle}")

        # Regla post (si existe)
        detalle_regla = ""
        if mot.strip():
            if 'NC / Ventas' in reporte_final.columns and pd.notna(reporte_final.loc[i,'NC / Ventas']):
                extra_p95 = f", p95={_fmt_pct__ripley(p95_ncv)}" if pd.notna(p95_ncv) else ""
                detalle_regla = f"; ajuste por regla ({mot}, NC/Ventas={_fmt_pct__ripley(reporte_final.loc[i,'NC / Ventas'])}{extra_p95})"
            else:
                detalle_regla = f"; ajuste por regla ({mot})"

        if len(frases) == 0: txt = "Catalogada por: sin drivers destacados (ver regla si aplica)"
        elif len(frases) == 1: txt = f"Catalogada por: {frases[0]}"
        else: txt = "Catalogada por: " + ", ".join(frases[:-1]) + " y " + frases[-1]

        textos.append(txt + detalle_regla)

    return pd.Series(textos, index=reporte_base.index)
COORDS_SUC = {
    "antofagasta": (-23.6500, -70.4000),
    "arauco maipu": (-33.5021, -70.7561),
    "arica": (-18.4783, -70.3126),
    "castellon": (-36.82699, -73.04977),
    "chillan": (-36.6066, -72.1034),
    "coquimbo": (-29.9533, -71.3436),
    "costanera center": (-33.4179, -70.6066),
    "costanera pto montt": (-41.4693, -72.9424),
    "crillon": (-33.4410, -70.6480),
    "el trebol": (-36.7901, -73.0867),
    "florida center": (-33.5014, -70.5989),
    "iquique": (-20.2133, -70.1517),
    "la calera": (-32.7870, -71.1950),
    "la serena": (-29.9027, -71.2519),
    "los andes": (-32.8330, -70.5980),
    "los dominicos": (-33.4085, -70.5515),
    "mall concepcion": (-36.82698, -73.04955),
    "mall curico": (-34.9837, -71.2384),
    "marina arauco": (-33.0245, -71.5516),
    "parque arauco": (-33.4019, -70.5724),
    "plaza alameda": (-33.4564, -70.6636),
    "plaza calama": (-22.4563, -68.9235),
    "plaza copiapo": (-27.3668, -70.3311),
    "plaza egana": (-33.4562, -70.5652),
    "plaza los angeles": (-37.4697, -72.3537),
    "plaza norte": (-33.3664, -70.6581),
    "plaza oeste": (-33.4957, -70.7078),
    "plaza sur": (-33.5923, -70.7041),
    "plaza tobalaba": (-33.5129, -70.5791),
    "plaza vespucio": (-33.5229, -70.5977),
    "portal temuco (mall temuco)": (-38.7393, -72.5987),
    "portal temuco": (-38.7393, -72.5987),
    "puente": (-33.6073, -70.5758),
    "puerto montt": (-41.4689, -72.9411),
    "punta arenas": (-53.1638, -70.9171),
    "quilpue": (-33.0450, -71.4429),
    "san fernando": (-34.5861, -70.9859),
    "talca": (-35.4290, -71.6618),
    "temuco": (-38.7359, -72.5904),
    "valdivia": (-39.8196, -73.2452),
    "valparaiso": (-33.0472, -71.6127),
    "vina del mar": (-33.0245, -71.5518),
    "vivo rancagua": (-34.1701, -70.7441),
}

def build_map_df(base_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    CAND_NAME = ['Nombre_Sucursal','sucursal','tienda','cod_sucursal','código_sucursal','codigo_sucursal','Sucursal']
    name_col = next((c for c in CAND_NAME if c in base_df.columns), None)
    if not name_col:
        raise KeyError("No encuentro columna de nombre de sucursal.")

    CAND_RISK = ['Riesgo_final','Cluster_Riesgo','Riesgo_P3_MinMax_Pesos']
    riesgo_col = next((c for c in CAND_RISK if c in base_df.columns), None)
    if not riesgo_col:
        raise KeyError("No encuentro columna de riesgo (Riesgo_final / Cluster_Riesgo / Riesgo_P3_MinMax_Pesos).")

    df_plot = base_df.rename(columns={name_col:'Nombre_Sucursal'}).copy()
    df_plot["__key"] = df_plot["Nombre_Sucursal"].map(_norm)
    df_plot["lat"] = df_plot["__key"].map(lambda k: COORDS_SUC.get(k, (np.nan,np.nan))[0])
    df_plot["lon"] = df_plot["__key"].map(lambda k: COORDS_SUC.get(k, (np.nan,np.nan))[1])

    missing = df_plot["lat"].isna() | df_plot["lon"].isna()
    if missing.any():
        print("⚠ Falta asignar coordenadas a estas sucursales (agrega la clave a COORDS_SUC):")
        print(df_plot.loc[missing, ["Nombre_Sucursal","__key"]].to_string(index=False))

    df_plot = df_plot.loc[~missing].copy()
    df_plot["lat"] = pd.to_numeric(df_plot["lat"], errors="coerce")
    df_plot["lon"] = pd.to_numeric(df_plot["lon"], errors="coerce")
    df_plot["Riesgo"] = df_plot[riesgo_col].astype(str)
    orden_riesgo = ["Riesgo Alto", "Riesgo Medio", "Riesgo Bajo"]
    colores = {"Riesgo Alto": "#e74c3c", "Riesgo Medio": "#f1c40f", "Riesgo Bajo": "#2ecc71"}
    presentes = [r for r in orden_riesgo if r in df_plot["Riesgo"].unique()]

    if "Productos_Dañados_Descuento" in df_plot.columns:
        s = pd.to_numeric(df_plot["Productos_Dañados_Descuento"], errors="coerce")
        df_plot["Prod_Danados_fmt"] = s.map(lambda v: f"${v:,.0f}" if pd.notna(v) else "s/d")
    else:
        df_plot["Prod_Danados_fmt"] = np.nan

    if "Total_Notas_Credito" in df_plot.columns:
        s = pd.to_numeric(df_plot["Total_Notas_Credito"], errors="coerce")
        df_plot["Total_NC_fmt"] = s.map(lambda v: f"${v:,.0f}" if pd.notna(v) else "s/d")
    else:
        df_plot["Total_NC_fmt"] = np.nan

    if "Desviacion_Meta_Merma" in df_plot.columns:
        s = pd.to_numeric(df_plot["Desviacion_Meta_Merma"], errors="coerce")
        df_plot["Desv_Meta_Merma_fmt"] = s.map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/d")
    else:
        df_plot["Desv_Meta_Merma_fmt"] = np.nan

    if "Costo_Merma_Destruccion" in df_plot.columns:
        s = pd.to_numeric(df_plot["Costo_Merma_Destruccion"], errors="coerce")
        df_plot["Costo_Merma_fmt"] = s.map(lambda v: f"${v:,.0f}" if pd.notna(v) else "s/d")
    else:
        df_plot["Costo_Merma_fmt"] = np.nan

    if "N° Multas" in df_plot.columns:
        s = pd.to_numeric(df_plot["N° Multas"], errors="coerce")
        df_plot["N_Multas_fmt"] = s.map(lambda v: f"{int(v)}" if pd.notna(v) else "n/d")
    else:
        df_plot["N_Multas_fmt"] = np.nan

    if "Total_Reclamos" in df_plot.columns:
        s = pd.to_numeric(df_plot["Total_Reclamos"], errors="coerce")
        df_plot["Total_Reclamos_fmt"] = s.map(lambda v: f"{int(v)}" if pd.notna(v) else "n/d")
    else:
        df_plot["Total_Reclamos_fmt"] = np.nan

    if "N° Accidentes" in df_plot.columns:
        s = pd.to_numeric(df_plot["N° Accidentes"], errors="coerce")
        df_plot["N_Accidentes_fmt"] = s.map(lambda v: f"{int(v)}" if pd.notna(v) else "n/d")
    else:
        df_plot["N_Accidentes_fmt"] = np.nan

    if "N° Recuperos 2025" in df_plot.columns:
        s = pd.to_numeric(df_plot["N° Recuperos 2025"], errors="coerce")
        df_plot["N_Recuperos_fmt"] = s.map(lambda v: f"{int(v)}" if pd.notna(v) else "n/d")
    else:
        df_plot["N_Recuperos_fmt"] = np.nan
   
   
   
    if "Detalle" not in df_plot.columns:
        df_plot["Detalle"] = ""

 

        # === NUEVO: usar 'Conclusiones' como Detalle del mapa ===
    try:
        if "Conclusiones" in df_plot.columns:
            # Mostramos conclusiones sólo para Riesgo Medio/Alto; en Bajo, lo dejamos vacío para no “ensuciar” el hover
            _mask_med_alto = df_plot["Riesgo"].astype(str).isin(["Riesgo Medio", "Riesgo Alto"])
            df_plot.loc[_mask_med_alto, "Detalle"] = df_plot.loc[_mask_med_alto, "Conclusiones"].fillna("").astype(str)
            df_plot.loc[~_mask_med_alto, "Detalle"] = ""
            # limpieza de espacios
            df_plot["Detalle"] = df_plot["Detalle"].str.replace(r"\s+", " ", regex=True).str.strip()
        else:
            print("⚠ 'Conclusiones' no está en df_plot; se mantiene la descripción manual.")
    except Exception as _e_detalle:
        print(f"⚠ No se pudo actualizar 'Detalle' con 'Conclusiones': {_e_detalle}")


    return df_plot, presentes, colores

def render_map_and_save(df_plot: pd.DataFrame, presentes: list[str], colores: dict, out_html: str) -> None:
    hover_cols = {"Riesgo": True, "Detalle": True, "lat": False, "lon": False}

    fig = px.scatter_mapbox(
        df_plot,
        lat="lat", lon="lon",
        color="Riesgo",
        hover_name="Nombre_Sucursal",
        hover_data=hover_cols,
        category_orders={"Riesgo": presentes},
        mapbox_style="carto-positron",
        zoom=4.8, center={"lat": -35.5, "lon": -71.5},
        color_discrete_map=colores,
        height=760
    )
    fig.update_traces(marker=dict(size=8, opacity=0.9))

    n_traces = len(fig.data)
    trace_names = [str(t.name) for t in fig.data]

    def vis_for_levels(selected_levels):
        if selected_levels == "Todos":
            return [True] * n_traces
        sel_low = [s.lower() for s in selected_levels]
        return [any(sl in tn.lower() for sl in sel_low) for tn in trace_names]

    buttons = [dict(label="Todos", method="update",
                    args=[{"visible": vis_for_levels("Todos")},
                          {"title": "Todos los niveles de riesgo"}])]
    for r in range(1, len(presentes)+1):
        for combo in combinations(presentes, r):
            buttons.append(dict(
                label=" & ".join(combo) if r > 1 else combo[0],
                method="update",
                args=[{"visible": vis_for_levels(combo)},
                      {"title": f"Sucursales - {' & '.join(combo)}"}]
            ))

    CONFIG = dict(
        scrollZoom=True,
        displaylogo=False,
        modeBarButtonsToRemove=["select2d","lasso2d","autoScale2d","toggleSpikelines"]
    )

    fig.update_layout(
        dragmode="zoom", uirevision="keep",
        mapbox=dict(center={"lat": -35.5, "lon": -71.5}, zoom=4.8, style="carto-positron"),
        legend=dict(title=dict(text="Nivel de Riesgo"), x=0, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=12, r=12, t=60, b=12),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            pad={"r":4,"t":4,"b":4,"l":4}, showactive=True,
            x=0.99, xanchor="right", y=0.99, yanchor="top",
            bgcolor="rgba(255,255,255,0.9)", borderwidth=1,
            font=dict(size=10), active=0, type="dropdown"
        )]
    )

    fig.write_html(out_html, include_plotlyjs="cdn", config=CONFIG)
    print(f"✅ Mapa generado: {out_html}")

def add_map_sheet_xlsxwriter(writer, map_html_path: str, sheet_name: str = "Mapa Interactivo"):
    workbook  = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    fmt_title = workbook.add_format({"bold": True, "font_size": 14})
    fmt_text  = workbook.add_format({"font_size": 11})
    fmt_link  = workbook.add_format({"font_color": "blue", "underline": 1})

    worksheet.write(0, 0, "Mapa Interactivo", fmt_title)
    worksheet.write(2, 0, "Abrir mapa en el navegador:", fmt_text)

    abs_html = os.path.abspath(map_html_path)
    worksheet.write_url(3, 0, "external:" + abs_html, fmt_link, string="Abrir mapa interactivo")
    worksheet.set_column(0, 0, 60)

def encontrar_nombre_oficial_mejorado(nombre, nombres_oficiales_normalizados, score_threshold=70):
    """
    Busca la mejor coincidencia para `nombre` dentro de `nombres_oficiales_normalizados`
    usando fuzzywuzzy.process.extractOne; si no hay lista, o el score es bajo, devuelve `nombre`.
    """
    try:
        if not nombres_oficiales_normalizados:
            return nombre
        # Garantizar que las opciones sean strings
        choices = [str(x) for x in nombres_oficiales_normalizados]
        res = process.extractOne(str(nombre), choices)
        if not res:
            return nombre
        cand, score = res
        try:
            score = float(score)
        except Exception:
            pass
        return cand if score >= score_threshold else nombre
    except Exception:
        # En caso de cualquier error, devolver el nombre original para no romper el flujo.
        return nombre

print("Creamos carpetas input y output si no existen.")
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

Detalle_sucursales = pd.read_excel(os.path.join(INPUT_DIR, "Detalle sucursales (1).xlsx"))


def procesar_base_sucursal(
    archivo_principal,
    detalle_sucursales=None,
    columna_id_principal=None,
    columna_id_detalle=None,
    columnas_a_sumar=None,
    mapeo_outlets=None,
    nombre_columna_sucursal='Tienda',
    columnas_seleccion=None,
    rename_dict=None,
    dropna_col=None,
    agrupacion=None,
    sheet_name=None,
    handle_productos_danados=False,
    handle_merma_destruccion=False,
    merma_cols=None,
    handle_multas=False,
    multas_sheet='JPL',
    multas_header=3,
    multas_nombre_col='Nombre_Sucursal',
    exclude_multas=None,
    multas_fecha_col=None,
    multas_year=None,
    handle_actividad_criminal=False,
    actividad_sheet='query',
    actividad_nombre_cols=None,
    manual_name_map=None,
    dropna_all=False,
    post_rename=None,
    fillna_0 = False,
    fillna_0_cols=None,
    collect_for_merge = False,
    exclude_names=None,
    exclude_regex=None,
    handle_rotacion=False,
    rotacion_sheet='Rot. Total',
    handle_reclamos=False,               
    reclamos_year_col='Año',             
    reclamos_ticket_col='Ticket',        
    reclamos_store_col='Tienda', 
    handle_juicios_laborales=False,
    juicios_sheet=None,
    juicios_header=None,
    juicios_nombre_col='División de Personal',
    juicios_fecha_col='Fecha Audiencia',
    juicios_year=None,

    ):
    """
    Procesamiento de bases de datos 
    - Ventas Tienda y Notas de Credito Tienda
    - Desviacion Inventario / Accidentes
    - Descuento Productos Dañados
    - Merma de Destruccion
    - Multas
    - Matriz BHT / Metros Cuadrado
    - Actividad Criminal (Recuperos)
    """
    global MERGE_DFS
    fname = str(archivo_principal)
    orig = fname
    tried = []

    if os.path.exists(orig):
        fname = orig
    else:
        if not os.path.isabs(orig):
            cand = os.path.join(INPUT_DIR, orig)
            tried.append(cand)
            if os.path.exists(cand):
                fname = cand
            else:
                cand2 = os.path.join(BASE_DIR, orig)
                tried.append(cand2)
                if os.path.exists(cand2):
                    fname = cand2
                else:
                    raise FileNotFoundError(f"No se encontró el archivo '{orig}'. Rutas intentadas: {tried}")
        else:
            raise FileNotFoundError(f"No se encontró ruta absoluta de '{orig}'.")    
    if fname.lower().endswith('.csv'):
        df = pd.read_csv(fname)
        print(f"Archivo CSV '{fname}' cargado. Registros: {len(df)}")
    elif fname.lower().endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')):
        # si se indicó sheet_name, forzar lectura de esa hoja; si no, leer la primer hoja por defecto
        try:
            if sheet_name is not None:
                df = pd.read_excel(fname, sheet_name=sheet_name)
                print(f"Archivo Excel '{fname}' cargado (hoja: '{sheet_name}'). Registros: {len(df)}")
            else:
                df = pd.read_excel(fname)
                print(f"Archivo Excel '{fname}' cargado (hoja por defecto). Registros: {len(df)}")
        except ValueError as e:
            # hoja no encontrada u otro error de sheet
            raise ValueError(f"No se pudo leer la hoja '{sheet_name}' en '{fname}': {e}")
    else:
        # intentamos leer como Excel por defecto si extensión desconocida
        df = pd.read_excel(fname, sheet_name=sheet_name) if sheet_name is not None else pd.read_excel(fname)
        print(f"Archivo '{fname}' cargado. Registros: {len(df)}")

    if dropna_all:
        n_before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"- dropna() aplicado a '{fname}': filas {n_before} -> {len(df)}")

    # ...existing code...
    # Merge con detalle de sucursales (si corresponde)
    if detalle_sucursales is not None and columna_id_principal and columna_id_detalle:
        df = pd.merge(
            left=df,
            right=detalle_sucursales,
            left_on=columna_id_principal,
            right_on=columna_id_detalle,
            how='left'
        )
        print("✅ Unión inicial completada.")

    # Flujo especial para 'Productos Dañados' 

    if handle_productos_danados or ('Código Sucursal' in df.columns and 'Descto' in df.columns):
        col_codigo = 'Código Sucursal'
        col_desc = 'Descto'
        print(f"\nAplicando flujo de 'Productos Dañados' (columnas detectadas: '{col_codigo}', '{col_desc}')...")

        # Extraer ID numérico del código de sucursal (prefijo de dígitos)
        df['Numero_Sucursal'] = df[col_codigo].astype(str).str.extract(r'^(\d+)')
        df['Numero_Sucursal'] = pd.to_numeric(df['Numero_Sucursal'], errors='coerce')
        df = df.dropna(subset=['Numero_Sucursal']).copy()
        df['Numero_Sucursal'] = df['Numero_Sucursal'].astype(int)
        print(f"- Se extrajeron {df['Numero_Sucursal'].nunique()} códigos de sucursal numéricos.")

        # Asegurar que 'Descto' sea numérico y agrupar por Numero_Sucursal
        df[col_desc] = pd.to_numeric(df[col_desc], errors='coerce').fillna(0)
        descuento_por_id = df.groupby('Numero_Sucursal').agg(Productos_Dañados_Descuento=(col_desc, 'sum')).reset_index()
        print(f"- Agrupado descuento por {len(descuento_por_id)} códigos de sucursal.")

        # Unir con detalle de sucursales si se proporcionó para obtener nombres
        if detalle_sucursales is not None:
            descuento_por_id = pd.merge(
                left=descuento_por_id,
                right=detalle_sucursales,
                left_on='Numero_Sucursal',
                right_on='Nº Suc.',
                how='left'
            )
            # Algunas hojas usan 'Tienda' como nombre; protegemos la existencia
            if 'Tienda' in descuento_por_id.columns:
                descuento_por_id = descuento_por_id.groupby('Tienda', as_index=False)['Productos_Dañados_Descuento'].sum()
                descuento_por_id = descuento_por_id.rename(columns={'Tienda': 'Nombre_Sucursal'})
                print("- Merge con Detalle_sucursales realizado y consolidado por 'Tienda'.")
            else:
                # si no hay columna Tienda, devolvemos por Numero_Sucursal
                descuento_por_id = descuento_por_id.rename(columns={'Numero_Sucursal': 'Numero_Sucursal_Id'})
                print("⚠ Detalle_sucursales no contenía columna 'Tienda'; se devuelve por ID de sucursal.")
        else:
            # si no hay detalle, devolvemos por Numero_Sucursal
            descuento_por_id = descuento_por_id.rename(columns={'Numero_Sucursal': 'Numero_Sucursal_Id'})
            print("⚠ No se proporcionó Detalle_sucursales; se devuelve resultado por ID de sucursal.")
        
        if collect_for_merge and 'Nombre_Sucursal' in descuento_por_id.columns:
            MERGE_DFS.append(descuento_por_id[['Nombre_Sucursal','Productos_Dañados_Descuento']].copy())

        return descuento_por_id
    
    # --- Flujo: Rotación de Personal (24-25) ---
    if handle_rotacion:
        try:
            rot_df = pd.read_excel(fname, sheet_name=rotacion_sheet, header=None)
        except Exception as e:
            raise FileNotFoundError(f"No se pudo leer hoja '{rotacion_sheet}' en '{fname}': {e}")

        # reconstruir headers y tabla a partir del layout dado
        headers = rot_df.iloc[1].astype(str).str.strip()
        rot_df = rot_df.iloc[2:].copy()
        rot_df.columns = headers
        rot_df = rot_df.dropna(axis=1, how="all")
        rot_df = rot_df.loc[:, ~rot_df.columns.duplicated()]

        def _pick(cols, opts):
            for c in cols:
                if str(c).strip().lower() in opts:
                    return c
            return None

        c_suc   = _pick(rot_df.columns, {"lugar de trabajo","sucursal","nombre_sucursal","tienda"})
        c_rot25 = "Rot. 2025" if "Rot. 2025" in rot_df.columns else next((c for c in rot_df.columns if "2025" in str(c)), None)

        if not (c_suc and c_rot25):
            raise ValueError("Rotación: faltan columnas requeridas (sucursal/2024/2025) en la hoja")

        Rotacion_Personal = rot_df[[c_suc, c_rot25]].copy()
        Rotacion_Personal.columns = ["Nombre_Sucursal", "Rot_2025"]
        Rotacion_Personal = Rotacion_Personal[Rotacion_Personal["Nombre_Sucursal"].astype(str).str.strip().str.lower() != "total"]

        def _parse_percent_column(series):
            s_raw = series.astype(str)
            had_percent = s_raw.str.contains("%")
            s_norm = (s_raw.str.replace("%","",regex=False)
                            .str.replace(",",".",regex=False)
                            .str.strip())
            vals = pd.to_numeric(s_norm, errors="coerce")
            vals = np.where(had_percent, vals/100.0, vals)
            mask_fix = (~had_percent) & pd.notna(vals) & (vals > 3)
            vals = np.where(mask_fix, vals/100.0, vals)
            return pd.to_numeric(vals, errors="coerce")

        raw25 = Rotacion_Personal["Rot_2025"].copy()
        Rotacion_Personal["Rot_2025"] = _parse_percent_column(raw25)

        Rotacion_Personal["Nombre_Sucursal"] = (
            Rotacion_Personal["Nombre_Sucursal"]
              .astype(str)
              .str.replace(r"^Tda\.\s*", "", regex=True)
              .str.strip()
        )

        _map_rotacion = map_manual_default.copy() if 'map_manual_default' in globals() else {}

        if 'encontrar_nombre_oficial_mejorado' in globals() and 'nombres_oficiales_normalizados' in globals():
            Rotacion_Personal["Nombre_Oficial_Match"] = Rotacion_Personal["Nombre_Sucursal"].apply(
                lambda x: encontrar_nombre_oficial_mejorado(x, nombres_oficiales_normalizados)
            )
        else:
            Rotacion_Personal["Nombre_Oficial_Match"] = Rotacion_Personal["Nombre_Sucursal"]

        Rotacion_Personal["Nombre_Oficial_Match"] = Rotacion_Personal.apply(
            lambda row: _map_rotacion.get(row["Nombre_Sucursal"], row["Nombre_Oficial_Match"]),
            axis=1
        )



        Rotacion_Personal["Nombre_Oficial_Match"] = Rotacion_Personal["Nombre_Oficial_Match"].fillna(Rotacion_Personal["Nombre_Sucursal"])
        Rotacion_Personal["Nombre_Sucursal"] = Rotacion_Personal["Nombre_Oficial_Match"]
        Rotacion_Personal = Rotacion_Personal.drop(columns=["Nombre_Oficial_Match"])

        if 'Nombre_Sucursal' in Rotacion_Personal.columns:
            # excluir por lista exacta (normalizando espacios)
            if exclude_names:
                excl_set = {str(x).strip() for x in exclude_names}
                Rotacion_Personal = Rotacion_Personal[
                    ~Rotacion_Personal["Nombre_Sucursal"].astype(str).str.strip().isin(excl_set)
                ].copy()
            # excluir por patrón regex
            if exclude_regex:
                Rotacion_Personal = Rotacion_Personal[
                    ~Rotacion_Personal["Nombre_Sucursal"].astype(str).str.contains(exclude_regex, na=False)
                ].copy()

        print(f"Rotacion_Personal procesada. Sucursales válidas: {Rotacion_Personal['Nombre_Sucursal'].nunique()}")

        if collect_for_merge:
            try:
                MERGE_DFS.append(Rotacion_Personal[['Nombre_Sucursal','Rot_2025']].drop_duplicates().copy())
                print("- Rotacion_Personal añadida a MERGE_DFS")
            except Exception as e:
                print("⚠ No se pudo añadir Rotacion_Personal a MERGE_DFS:", e)

        return Rotacion_Personal
    

    # --- Flujo: Reclamos 2025 ---
    if handle_reclamos or 'reclamos' in fname.lower():
        # Reclamos: soporta archivo con columnas año, tienda, ticket
        if reclamos_store_col not in df.columns and archivo_principal:
            pass

        # filtrar año 2025 (configurable)
        if reclamos_year_col in df.columns:
            recl_2025 = df[df[reclamos_year_col] == 2025].copy()
        else:
            recl_2025 = df.copy()

        # eliminar filas sin tienda
        if reclamos_store_col in recl_2025.columns:
            recl_2025 = recl_2025.dropna(subset=[reclamos_store_col]).reset_index(drop=True)
            recl_2025[reclamos_store_col] = recl_2025[reclamos_store_col].astype(str).str.strip()
        else:
            raise KeyError(f"No se encontró columna de tienda '{reclamos_store_col}' en archivo de reclamos.")

        # contar tickets por tienda
        if reclamos_ticket_col in recl_2025.columns:
            recl_sum = recl_2025.groupby(reclamos_store_col).agg(
                Reclamos_Cantidad_2025=(reclamos_ticket_col, 'count')
            ).reset_index()
        else:
            recl_sum = recl_2025.groupby(reclamos_store_col).size().reset_index(name='Reclamos_Cantidad_2025')

        recl_sum = recl_sum.rename(columns={reclamos_store_col: 'Nombre_Sucursal'})

        # normalizador simple (eliminar tildes)
        def _eliminar_tildes(texto):
            texto = str(texto)
            return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn').strip()

        # construir diccionario de correcciones: usar map_manual_default + manual_name_map (si existe)
        correcciones = {}
        if 'map_manual_default' in globals():
            correcciones.update(map_manual_default)
        if manual_name_map:
            correcciones.update(manual_name_map)

        # fixes explícitos adicionales para casos conocidos
        correcciones.update({
            "Concepcion 2": "Castellón",
            "Concepción 2": "Castellón",
        })

        # normalizar claves del diccionario (sin tildes, trimmed)
        correcciones_norm = { _eliminar_tildes(k).strip(): v for k, v in correcciones.items() }

        # aplicar mapping únicamente por coincidencia en diccionario normalizado
        def _map_nombre(nombre_orig):
            clave = _eliminar_tildes(nombre_orig).strip()
            if clave in correcciones_norm:
                return correcciones_norm[clave]
            return nombre_orig  # si no hay mapping, conservar original

        recl_sum['Nombre_Sucursal'] = recl_sum['Nombre_Sucursal'].astype(str).apply(lambda x: _map_nombre(x).strip())

        # eliminar filas no deseadas explícitas (ej: "Concepcion 1" si aplica)
        try:
            recl_sum = recl_sum[recl_sum['Nombre_Sucursal'].astype(str).str.strip().astype(bool)].copy()
        except Exception:
            pass

        # agrupar por nombre final y consolidar cantidades
        recl_final = recl_sum.groupby('Nombre_Sucursal', as_index=False).agg({'Reclamos_Cantidad_2025': 'sum'})

        # excluir por lista o regex si se pidió
        if 'Nombre_Sucursal' in recl_final.columns:
            if exclude_names:
                excl_set = {str(x).strip() for x in exclude_names}
                recl_final = recl_final[~recl_final['Nombre_Sucursal'].astype(str).str.strip().isin(excl_set)].copy()
            if exclude_regex:
                recl_final = recl_final[~recl_final['Nombre_Sucursal'].astype(str).str.contains(exclude_regex, na=False)].copy()

        print(f"Reclamos procesados. Sucursales con reclamos 2025: {len(recl_final)}")

        # añadir a MERGE_DFS si se solicitó
        if collect_for_merge and 'Nombre_Sucursal' in recl_final.columns:
            MERGE_DFS.append(recl_final[['Nombre_Sucursal','Reclamos_Cantidad_2025']].copy())
            print("- Reclamos añadidos a MERGE_DFS")

        return recl_final
# ...existing code...


 # --- Flujo: Merma de Destrucción (celda 20) ---
    # Detecta si la tabla parece merma o forzar con handle_merma_destruccion=True
    if handle_merma_destruccion or ('Sucursal' in df.columns and (('Costo_Merma' in df.columns) or ('Unidades_Merma' in df.columns))):
        # configuración por defecto de nombres de columnas (se puede pasar merma_cols)
        mc = {'sucursal_col': 'Sucursal', 'costo_col': 'Costo_Merma', 'unidades_col': 'Unidades_Merma'}
        if isinstance(merma_cols, dict):
            mc.update(merma_cols)

        s_col = mc['sucursal_col']
        costo_col = mc['costo_col']
        unidades_col = mc['unidades_col']

        print(f"Aplicando flujo 'Merma de Destrucción' (sucursal='{s_col}', costo='{costo_col}', unidades='{unidades_col}')")

        # limpiar la columna de sucursal (eliminar prefijo '100' si existe)
        if s_col not in df.columns:
            raise KeyError(f"No se encontró la columna esperada de sucursal: '{s_col}'")
        df['ID_Sucursal_Limpio'] = df[s_col].astype(str).str.replace(r'^100', '', regex=True)
        df['ID_Sucursal_Limpio'] = pd.to_numeric(df['ID_Sucursal_Limpio'], errors='coerce')
        df = df.dropna(subset=['ID_Sucursal_Limpio']).copy()
        df['ID_Sucursal_Limpio'] = df['ID_Sucursal_Limpio'].astype(int)
        print(f"- Se obtuvieron {df['ID_Sucursal_Limpio'].nunique()} IDs de sucursal numéricos.")

        # asegurar columnas numéricas
        if costo_col in df.columns:
            df[costo_col] = pd.to_numeric(df[costo_col], errors='coerce').fillna(0)
        else:
            df[costo_col] = 0
            print(f"⚠ No se encontró columna '{costo_col}', se usará 0.")

        if unidades_col in df.columns:
            df[unidades_col] = pd.to_numeric(df[unidades_col], errors='coerce').fillna(0)
        else:
            df[unidades_col] = 0
            print(f"⚠ No se encontró columna '{unidades_col}', se usará 0.")

        # agrupar por ID limpio
        merma_agg = df.groupby('ID_Sucursal_Limpio').agg(
            Merma_Destruccion_Costo=(costo_col, 'sum'),
            Merma_Destruccion_Unidades=(unidades_col, 'sum')
        ).reset_index()
        print(f"- Agrupadas merma para {len(merma_agg)} sucursales (por ID).")

        # unir con detalle_sucursales para obtener nombre de tienda si se entregó
        if detalle_sucursales is not None:
            merma_agg = pd.merge(merma_agg, detalle_sucursales, left_on='ID_Sucursal_Limpio', right_on='Nº Suc.', how='left')
            if 'Tienda' in merma_agg.columns:
                merma_agg['Nombre_Sucursal'] = merma_agg['Tienda'].astype(str).str.strip()
                mask_no_match = (
                    merma_agg['Nombre_Sucursal'].isnull() |
                    merma_agg['Nombre_Sucursal'].str.strip().str.lower().isin(['', 'nan', 'none'])
                )
                n_no_match = int(mask_no_match.sum())
                if n_no_match > 0:
                    merma_agg = merma_agg.loc[~mask_no_match].copy()
                    print(f"- Se eliminaron {n_no_match} filas sin correspondencia en Detalle_sucursales (ID sin match).")

                # aplicar mapeo outlets si se entregó
                if mapeo_outlets:
                    merma_agg['Nombre_Sucursal'] = merma_agg['Nombre_Sucursal'].replace(mapeo_outlets)
                merma_agg = merma_agg.groupby('Nombre_Sucursal')[['Merma_Destruccion_Costo', 'Merma_Destruccion_Unidades']].sum().reset_index()
                print("- Merge con Detalle_sucursales realizado y consolidado por 'Tienda'.")
            else:
                # si no hay columna 'Tienda' dejamos por ID
                merma_agg = merma_agg.rename(columns={'ID_Sucursal_Limpio': 'ID_Sucursal_Limpio_Id'})
                print("⚠ Detalle_sucursales no contenía columna 'Tienda'; resultado devuelto por ID de sucursal.")
        else:
            merma_agg = merma_agg.rename(columns={'ID_Sucursal_Limpio': 'ID_Sucursal_Limpio_Id'})
            print("⚠ No se proporcionó Detalle_sucursales; resultado devuelto por ID de sucursal.")

        merma_agg = merma_agg.rename(columns={
            'Merma_Destruccion_Costo': 'Costo_Merma_Destruccion',
            'Merma_Destruccion_Unidades': 'Merma_Destruccion_Unidades'
        })

        if collect_for_merge and 'Nombre_Sucursal' in merma_agg.columns:
            MERGE_DFS.append(merma_agg[['Nombre_Sucursal','Costo_Merma_Destruccion','Merma_Destruccion_Unidades']].copy())

        return merma_agg
    
     # --- Flujo: Multas (celda 22) ---
    # --- Flujo: Multas (celda 22) ---
    if handle_multas or fname.lower().startswith('202506 informe de juicios'):            # lee hoja y header configurables (por defecto JPL, header=3)
            try:
                multas_df = pd.read_excel(fname, sheet_name=multas_sheet, header=multas_header)
            except Exception:
                # fallback a lectura sin header si falla
                multas_df = pd.read_excel(fname, sheet_name=multas_sheet)
            print(f"Archivo de Multas '{fname}' (hoja: {multas_sheet}) cargado con {len(multas_df)} registros.")
    
            # columna con nombre de sucursal (por defecto "Nombre_Sucursal")
            if multas_nombre_col not in multas_df.columns:
                # intentar variantes comunes (coincidencia exacta ignorando mayúsculas/minúsculas)
                found_col = None
                candidates = ['Nombre_Sucursal', 'Nombre Sucursal', 'Nombre', 'Sucursal', 'Tienda', 'División de Personal', 'Division de Personal']
                for c_try in candidates:
                    for col in multas_df.columns:
                        try:
                            if str(col).strip().casefold() == str(c_try).strip().casefold():
                                found_col = col
                                break
                        except Exception:
                            continue
                    if found_col:
                        break
    
                # si no se encontró con igualdad, intentar buscar por palabras clave contenidas
                if found_col is None:
                    keywords = ['nombre', 'sucursal', 'tienda', 'división', 'division']
                    for col in multas_df.columns:
                        col_str = str(col)
                        if any(k in col_str.casefold() for k in keywords):
                            found_col = col
                            break
    
                if found_col is None:
                    raise KeyError(f"No se encontró columna de nombre de sucursal en archivo de multas (intentadas incl.: {multas_nombre_col}).")
                multas_nombre_col = found_col
    
            print(f"Usando columna de nombre para Multas: '{multas_nombre_col}'")
    
            # Si la columna detectada es tipo "División de Personal" (o contiene 'division'), contar filas por esa columna
            try:
                col_lower = str(multas_nombre_col).casefold()
            except Exception:
                col_lower = ''
    
            if 'division' in col_lower or 'división' in col_lower:
                # normalizar valores, eliminar blancos y nulos, contar ocurrencias
                s = multas_df[multas_nombre_col].astype(str).str.strip()
                s = s.replace({'nan': None, 'none': None, '': None})
                cnt = s.dropna().value_counts().reset_index()
                cnt.columns = ['Nombre_Sucursal', 'N° Multas']
                cnt['N° Multas'] = cnt['N° Multas'].astype(int)
                print(f"Conteo por '{multas_nombre_col}' realizado. Sucursales con registros: {len(cnt)}")
                if collect_for_merge:
                    MERGE_DFS.append(cnt.copy())
                    print("- Conteo de Multas por División añadido a MERGE_DFS")
                return cnt
    
            # conteo por sucursal
            Multas_Sucursal = multas_df[multas_nombre_col].value_counts().reset_index()
            Multas_Sucursal.columns = ['Nombre_Sucursal', 'N° Multas']
    
            # Excluir sucursales no deseadas (Marketplace, Barros Arana, etc.)
            if exclude_multas is None:
                exclude_multas = ["Marketplace", "Barros Arana"]
            Multas_Sucursal = Multas_Sucursal[~Multas_Sucursal['Nombre_Sucursal'].isin(exclude_multas)].reset_index(drop=True)
    
            # consolidar (suma por nombre)
            Multas_Sucursal = Multas_Sucursal.groupby('Nombre_Sucursal', as_index=False)['N° Multas'].sum()
            Multas_Sucursal['N° Multas'] = pd.to_numeric(Multas_Sucursal['N° Multas'], errors='coerce').fillna(0).astype(int)
    
            print(f"Se agruparon las multas para {len(Multas_Sucursal)} sucursales con registros.")
    
            if collect_for_merge and 'Nombre_Sucursal' in Multas_Sucursal.columns:
                MERGE_DFS.append(Multas_Sucursal[['Nombre_Sucursal','N° Multas']].copy())
    
            return Multas_Sucursal
        # --- Flujo: Juicios Laborales (conteo por División de Personal, filtrable por año de Fecha Audiencia) ---
    # --- Flujo: Juicios Laborales (conteo por División de Personal, filtrable por año de Fecha Audiencia) ---
    if handle_juicios_laborales:
        # 1) Lectura controlada
        try:
            jl_df = pd.read_excel(fname, sheet_name=juicios_sheet, header=juicios_header)
        except Exception:
            jl_df = pd.read_excel(fname)

        print(f"Juicios Laborales: '{fname}' (hoja={juicios_sheet}, header={juicios_header}) cargado con {len(jl_df)} registros.")

        # 2) Detectar columna de división si no viene exacta
        if juicios_nombre_col not in jl_df.columns:
            def _norm_col(s):
                s = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
                return re.sub(r'\s+', ' ', s).strip().casefold()
            norm_map = {_norm_col(c): c for c in jl_df.columns}
            candidatos = ['division de personal','division personal','división de personal','división personal','division']
            hit = None
            for k in candidatos:
                if k in norm_map:
                    hit = norm_map[k]; break
            if not hit:
                for k, orig in norm_map.items():
                    if 'division' in k and ('personal' in k or 'persona' in k or 'perso' in k):
                        hit = orig; break
            if hit:
                juicios_nombre_col = hit
        if juicios_nombre_col not in jl_df.columns:
            raise KeyError(f"Juicios Laborales: no se encontró columna de división (intentado: '{juicios_nombre_col}'). Columnas: {list(jl_df.columns)[:30]}")

        # 3) Filtro por año usando 'Fecha Audiencia'
        if juicios_year is not None:
            col_fecha = juicios_fecha_col if juicios_fecha_col in jl_df.columns else None
            if col_fecha is None:
                def _norm_txt(s):
                    s = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
                    return re.sub(r'\s+', ' ', s).strip().casefold()
                for c in jl_df.columns:
                    n = _norm_txt(c)
                    if ('fecha' in n) and ('audien' in n):
                        col_fecha = c; break
                if col_fecha is None:
                    for c in jl_df.columns:
                        try:
                            _ = pd.to_datetime(jl_df[c].dropna().iloc[:10], errors='raise', dayfirst=True)
                            col_fecha = c; break
                        except Exception:
                            continue
            if col_fecha:
                fechas = pd.to_datetime(jl_df[col_fecha], errors='coerce', dayfirst=True, infer_datetime_format=True)
                jl_df = jl_df.loc[fechas.dt.year == int(juicios_year)].copy()
                print(f"- Filtro año aplicado sobre '{col_fecha}': {int(juicios_year)} -> filas {len(jl_df)}")
            else:
                print("⚠ Juicios Laborales: no se detectó columna de fecha; no se aplicó filtro por año.")

        # 4) Limpieza y exclusiones en División/Nombre_Sucursal
        excl_raw = {
            "Ag 21 de Mayo","Ag Rancagua","Back Office","Contact Center","Estado 91","Externo",
            "Huérfanos","Of Estado 91","Of Miraflores","Reservada","Tda Alto las Condes","VARIAS", "nan"
        }
        def _norm_val(x):
            s = ''.join(c for c in unicodedata.normalize('NFD', str(x)) if unicodedata.category(c) != 'Mn')
            s = re.sub(r'\s+', ' ', s).strip()
            s = re.sub(r'(?i)^\s*(tda\.?\s*|cd\.?\s*)', '', s)  # quita prefijos al inicio
            return s.casefold()

        excl_norm = {_norm_val(v) for v in excl_raw}

        col_div = juicios_nombre_col

        # quitar prefijos "Tda"/"CD" al inicio de los valores de la columna y limpiar espacios
        jl_df[col_div] = (
            jl_df[col_div]
            .astype(str)
            .str.replace(r'(?i)^\s*(tda\.?\s*|cd\.?\s*)', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

        # excluir vacíos
        jl_df = jl_df[jl_df[col_div].notna() & (jl_df[col_div].str.len() > 0)].copy()

        # excluir lista indicada (comparación robusta tras normalizar y quitar prefijos)
        jl_df = jl_df[~jl_df[col_div].apply(lambda v: _norm_val(v) in excl_norm)].copy()

        # 5) Mapeo manual (usa el global map_manual_default) + opcional manual_name_map
        base_map = globals().get('map_manual_default', {})
        if base_map:
            jl_df[col_div] = jl_df[col_div].replace(base_map)
        if manual_name_map:
            jl_df[col_div] = jl_df[col_div].replace(manual_name_map)

        # 6) Conteo y renombre a 'Nombre_Sucursal'
        out_col = f"Juicios_Laborales" if juicios_year is not None else "Juicios_Laborales"
        out = (
            jl_df.groupby(col_div, dropna=False)
                 .size()
                 .reset_index(name=out_col)
            .rename(columns={col_div: "Nombre_Sucursal"})
        )

        # 7) Integración opcional al MERGE_DFS (para merge maestro por 'Nombre_Sucursal')
        if collect_for_merge and 'Nombre_Sucursal' in out.columns:
            MERGE_DFS.append(out[['Nombre_Sucursal', out_col]].copy())
            print(f"- Juicios Laborales añadidos a MERGE_DFS (columna '{out_col}')")

        print(f"Juicios Laborales: conteo realizado. Filas resultado: {len(out)}")
        return out


        
        # Flujo: Actividad Criminal (Recuperos)

    if handle_actividad_criminal or ('actividad criminal' in fname.lower()) or ('recuperos' in fname.lower()):
        ac_file_candidates = []
        if 'actividad criminal' in fname.lower() or 'recuperos' in fname.lower():
            ac_file_candidates.append(fname)
        ac_file_candidates.append(os.path.join(os.getcwd(), "Actividad Criminal.xlsx"))
        ac_file_candidates.append("Actividad Criminal.xlsx")

        act_df = None
        for af in ac_file_candidates:
            try:
                act_df = pd.read_excel(af, sheet_name=actividad_sheet) if actividad_sheet is not None else pd.read_excel(af)
                print(f"Actividad Criminal: archivo '{af}' cargado. Registros: {len(act_df)}")
                break
            except FileNotFoundError:
                continue
            except Exception:
                continue
        if act_df is None:
            raise FileNotFoundError("No se encontró 'Actividad Criminal.xlsx' ni el archivo indicado en archivo_principal.")

        # detectar columna de fecha y preparar año
        fecha_col = 'Fecha_Evento' if 'Fecha_Evento' in act_df.columns else None
        if fecha_col is None:
            for c in act_df.columns:
                try:
                    _ = pd.to_datetime(act_df[c].dropna().iloc[:10], dayfirst=True, errors='raise')
                    fecha_col = c
                    break
                except Exception:
                    continue
        if fecha_col:
            act_df['Fecha_Evento'] = pd.to_datetime(act_df[fecha_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
        else:
            act_df['Fecha_Evento'] = pd.NaT

        act_df['Año'] = act_df['Fecha_Evento'].dt.year
        act_2025 = act_df[act_df['Año'] == 2025].copy()

        # columna de nombre (permitir variantes)
        name_col = None
        if actividad_nombre_cols:
            for c in actividad_nombre_cols:
                if c in act_2025.columns:
                    name_col = c
                    break
        if name_col is None:
            for c_try in ['Título', 'Nombre_Sucursal', 'Nombre Sucursal', 'Sucursal', 'Tienda']:
                if c_try in act_2025.columns:
                    name_col = c_try
                    break
        if name_col is None:
            raise KeyError("No se encontró columna de nombre de sucursal en 'Actividad Criminal' (se buscó 'Título' u otras variantes).")

        resumen_act = (
            act_2025
            .groupby(name_col, dropna=False)
            .size()
            .reset_index(name='N° Recuperos 2025')
        ).rename(columns={name_col: 'Nombre_Sucursal'})

        # limpieza básica de nombres
        resumen_act['Nombre_Sucursal'] = (
            resumen_act['Nombre_Sucursal'].astype(str)
            .str.replace(r'(?i)^\s*Tienda[\s\-\–:]*', '', regex=True)
            .str.replace(r'(?i)\btda\.?\b', '', regex=True)
            .str.replace(r'(?i)^\s*ripley[\s\-\–:]*', '', regex=True)
            .str.replace(r'(?i)^\s*Comercial /[\s\-\–:]*', '', regex=True)
            .str.replace(r'^\s*\.+\s*', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

        # mapeos manuales por defecto (incluye tus correcciones comunes)
        local_map_manual_default = {
            'Mall Plaza Trebol': 'El Trébol',
            'Plaza Huechuraba': 'Plaza Norte',
            'Concepcion Castellon': 'Castellón',
            'Pto. Montt Costanera': 'Costanera Pto Montt',
            'Copiapó': 'Plaza Copiapó',
            'Curicó': 'Mall Curicó',
            'Pta Arenas': 'Punta Arenas',
            'San Bernardo': 'Plaza Sur',
            'Mall del Centro Rancagua': 'Vivo Rancagua',
            'Rancagua': 'Vivo Rancagua',
            'Mall Del Centro': 'Puente',
            'Viña Sucre': 'Viña del Mar',
            'Concepcion': 'Castellón',
            'Mall Castellón': 'Castellón',
            'Maipu': 'Arauco Maipú',
            'Maipú': 'Arauco Maipú',
            'Los Angeles': 'Plaza Los Ángeles',
            'Mall Calama': 'Plaza Calama',
            'Nueva Valdivia': 'Valdivia',
            'Mall Costanera': 'Costanera Pto Montt',
            'Mall Plaza Alameda': 'Plaza Alameda',
            "Mall Temuco": "Portal Temuco (Mall Temuco)",
            "Marina Arauco - Seguridad": "Marina Arauco",
            "Chillan": "Chillán",
            "Crillon": "Crillón",
            "Mall Concepcion": "Mall Concepción",
            "Quilpue": "Quilpué",
            "Valparaiso": "Valparaíso",
            "Arauco Maipu": "Arauco Maipú",
            # entradas adicionales menores si se requieren pueden añadirse via manual_name_map
        }

        # combinar mapeos: si existe un map_manual_default a nivel global úsalo como base,
        # luego aplicar los locales, y por último el manual_name_map pasado por el caller.
        base_map = globals().get('map_manual_default', {})
        merged_map = {}
        merged_map.update(base_map)
        merged_map.update(local_map_manual_default)

        # aplicar mapeos: merged_map primero, luego manual_name_map si se entregó
        if merged_map:
            resumen_act['Nombre_Sucursal'] = resumen_act['Nombre_Sucursal'].replace(merged_map)
        if manual_name_map:
            resumen_act['Nombre_Sucursal'] = resumen_act['Nombre_Sucursal'].replace(manual_name_map)

        resumen_act = resumen_act.groupby('Nombre_Sucursal', as_index=False)['N° Recuperos 2025'].sum()
        resumen_act['N° Recuperos 2025'] = pd.to_numeric(resumen_act['N° Recuperos 2025'], errors='coerce').fillna(0).astype(int)

        resumen_act = resumen_act[~resumen_act['Nombre_Sucursal'].astype(str).str.strip().str.casefold().isin({'alto las condes','mall alto las condes','nan', "ofi cerro colorado 5240"})].reset_index(drop=True)

        print(f"Actividad Criminal: {len(resumen_act)} sucursales con recuperos en 2025 procesadas.")
        
        if collect_for_merge and 'Nombre_Sucursal' in resumen_act.columns:
            MERGE_DFS.append(resumen_act[['Nombre_Sucursal','N° Recuperos 2025']].copy())
        return resumen_act


    # Eliminar nulos en columna clave (si corresponde)
    if dropna_col:
        n_nulos = df[dropna_col].isnull().sum()
        if n_nulos > 0:
            df.dropna(subset=[dropna_col], inplace=True)
            print(f"- Se eliminaron {n_nulos} filas sin valor en '{dropna_col}'.")

    # Unificar nombres de outlets (si corresponde)
    if mapeo_outlets and nombre_columna_sucursal in df.columns:
        df[nombre_columna_sucursal] = df[nombre_columna_sucursal].replace(mapeo_outlets)
        print("- Se unificaron los nombres de las sucursales Outlet con sus tiendas principales.")

    # Seleccionar columnas (si corresponde)
    if columnas_seleccion:
        df = df[columnas_seleccion].copy()

    # Renombrar columnas (si corresponde)
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print("- Se renombraron las columnas según el diccionario.")

    # Agrupación y suma (si corresponde)
    if agrupacion and columnas_a_sumar:
        df = df.groupby(agrupacion)[columnas_a_sumar].sum().reset_index()
        print("- Consolidación por sucursal completada.")

    if post_rename:
        try:
            df = df.rename(columns=post_rename)
            print(f"- post_rename aplicado: {list(post_rename.keys())} -> {[post_rename[k] for k in post_rename]}")
        except Exception as e:
            print(f"⚠ Error aplicando post_rename: {e}")
    
    if fillna_0:
        if fillna_0_cols:
            cols_to_fill = [c for c in fillna_0_cols if c in df.columns]
            if cols_to_fill:
                df[cols_to_fill] = df[cols_to_fill].fillna(0)
                print(f"- fillna(0) aplicado a columnas: {cols_to_fill}")
            else:
                print("⚠ fillna_0_cols no contiene columnas existentes en el DataFrame; no se aplicó fillna.")
        else:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                df[num_cols] = df[num_cols].fillna(0)
    if collect_for_merge:
        try:
            # asegurar Nombre_Sucursal disponible
            if 'Nombre_Sucursal' not in df.columns and nombre_columna_sucursal in df.columns:
                df = df.rename(columns={nombre_columna_sucursal: 'Nombre_Sucursal'})
            if 'Nombre_Sucursal' not in df.columns:
                print("⚠ collect_for_merge solicitado pero no se encontró 'Nombre_Sucursal' en el DataFrame; no se añadió.")
            else:
                MERGE_DFS.append(df.copy())
                print(f"- DataFrame añadido a MERGE_DFS (filas: {len(df)} columnas: {len(df.columns)})")
        except Exception as e:
            print(f"⚠ Error añadiendo a MERGE_DFS: {e}")
    
    if 'Nombre_Sucursal' in df.columns:
        if exclude_names:
            excl_set = set(exclude_names)  # coincidencia exacta
            df = df[~df["Nombre_Sucursal"].isin(excl_set)]
        if exclude_regex:
            df = df[~df["Nombre_Sucursal"].astype(str).str.contains(
            exclude_regex, na=False  # quita filas que hagan match
        )]
    return df
# ...existing code...
mapeo_outlets = {
    'Outlet La Serena': 'La Serena',
    'Outlet Puerto Montt Costanera': 'Costanera Pto Montt',
    'Outlet Puente': 'Puente',
    'Outlet Iquique': 'Iquique',
    'Outlet Parque Arauco': 'Parque Arauco'
}

# Mapeo por defecto usado en el flujo de "Actividad Criminal".
# Se define a nivel de notebook para poder pasarlo como argumento a la función.
map_manual_default = {
    'Mall Plaza Trebol': 'El Trébol',
    'Plaza Huechuraba': 'Plaza Norte',
    "Huechuraba": "Plaza Norte",
    'Concepcion Castellon': 'Castellón',
    'Pto. Montt Costanera': 'Costanera Pto Montt',
    "Puerto Montt Costanera": "Costanera Pto Montt",
    'Copiapó': 'Plaza Copiapó',
    'Curicó': 'Mall Curicó',
    'Pta Arenas': 'Punta Arenas',
    'San Bernardo': 'Plaza Sur',
    'Mall del Centro Rancagua': 'Vivo Rancagua',
    'Rancagua': 'Vivo Rancagua',
    'Mall Del Centro': 'Puente',
    "Mall del Centro": "Puente",
    "Mall del centro" : "Puente",
    'Viña Sucre': 'Viña del Mar',
    "Viña Plaza Sucre": "Viña del Mar",
    'Concepcion': 'Castellón',
    'Mall Castellón': 'Castellón',
    'Maipu': 'Arauco Maipú',
    "Maipú": 'Arauco Maipú',
    "Concepción Castellón": "Castellón",
    "Plaza El Trébol": "El Trébol",
    "Trebol": "El Trébol",
    "Puerto Montt Centro" : "Puerto Montt",
    "Alameda" : "Plaza Alameda",
    "los Angeles" : "Plaza Los Ángeles",
    'Los Angeles': 'Plaza Los Ángeles',
    "Los Ángeles": "Plaza Los Ángeles",
    'Mall Calama': 'Plaza Calama',
    'Nueva Valdivia': 'Valdivia',
    'Mall Costanera': 'Costanera Pto Montt',
    'Mall Plaza Alameda': 'Plaza Alameda',
    "Mall Temuco": "Portal Temuco (Mall Temuco)",
    "Marina Arauco - Seguridad": "Marina Arauco",
    "Chillan": "Chillán",
    "Crillon": "Crillón",
    "Calama" : "Plaza Calama",
    "Portal Temuco": "Portal Temuco (Mall Temuco)",
    "Mall Concepcion": "Mall Concepción",
    "Quilpue": "Quilpué",
    "Valparaiso": "Valparaíso",
    "Arauco Maipu": "Arauco Maipú",
    "Estación Central": "Plaza Alameda",
    "Estacion Central": "Plaza Alameda",
    "Plaza El Trebol" : "El Trébol",
    "Curico" : "Mall Curicó",

    # entradas adicionales menores si se requieren pueden añadirse via manual_name_map
}

MERGE_DFS = []

def merge_all_on_nombre_sucursal(dfs, how='outer', fillna_zero=False):
    if not dfs:
        print("MERGE_DFS vacío: no hay DataFrames para mergear.")
        return pd.DataFrame()
    merged = dfs[0].copy()
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on='Nombre_Sucursal', how=how)
    if fillna_zero:
        num_cols = merged.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            merged[num_cols] = merged[num_cols].fillna(0)
    return merged

def autofit_columns(worksheet, df, index = False, min_width=8, extra = 2):
    start_col = 0
    if index:
        idx_vals_max = df.index.map(lambda x: len(str(x))).max() if len(df.index) else 0
        idx_hdr_len = len(str(df.index.name or ""))
        idx_width = max(idx_vals_max, idx_hdr_len, min_width) + extra
        worksheet.set_column(0, 0, idx_width)
        start_col = 1
    for i, col in enumerate(df.columns, start = start_col):
        series = df[col].astype(str) if len(df) else pd.Series(dtype=str)
        vals_max = series.map(len).max() if len(series) else 0
        hdr_len = len(str(col))
        width = max(vals_max, hdr_len, min_width) + extra
        worksheet.set_column(i, i, width) 

Detalle_sucursales = pd.read_excel(os.path.join(INPUT_DIR, "Detalle sucursales (1).xlsx"))

Ventas_sucusal_2025 = procesar_base_sucursal(
    archivo_principal="Ventas_Sucursal_2025.csv",
    detalle_sucursales=Detalle_sucursales,
    columna_id_principal='sucursal',
    columna_id_detalle='Nº Suc.',
    columnas_a_sumar=['total_monto_trx', 'total_transacciones'],
    mapeo_outlets=mapeo_outlets,
    nombre_columna_sucursal='Tienda',
    agrupacion='Tienda',
    post_rename={
        'Tienda': 'Nombre_Sucursal', 
        'total_monto_trx': 'Total_Ventas',
        'total_transacciones': 'Total_Transacciones_Ventas'},
    collect_for_merge=True
)

# Ejemplo para Notas de Crédito
Notas_Credito_Sucursal_2025 = procesar_base_sucursal(
    archivo_principal="Notas_Credito_Sucursal_2025.csv",
    detalle_sucursales=Detalle_sucursales,
    columna_id_principal='sucursal',
    columna_id_detalle='Nº Suc.',
    columnas_a_sumar=['total_monto_trx', 'total_transacciones'],
    mapeo_outlets=mapeo_outlets,
    nombre_columna_sucursal='Tienda',
    agrupacion='Tienda',
        post_rename={
        'Tienda': 'Nombre_Sucursal', 
        'total_monto_trx': 'Total_Notas_Credito',
        'total_transacciones': 'Total_Transacciones_Notas_Credito'},
    collect_for_merge=True

)

# Ejemplo para Merma de Inventario
Desviacion_Meta_Merma_2025 = procesar_base_sucursal(
    archivo_principal="Indice_Meta_Merma_Inventario_2024.xlsx",
    sheet_name="2025",
    columnas_seleccion=['Tienda', 'Desviacion_Meta_Merma'],
    rename_dict={'Tienda': 'Nombre_Sucursal'},
    dropna_col='Desviacion_Meta_Merma',
    collect_for_merge=True
)

Accidentes_2025 = procesar_base_sucursal(
    archivo_principal="Accidentes_Sucursal_2024.xlsx",
    sheet_name="2025",                                
    columnas_seleccion=['Tienda', 'N° Accidentes'],   
    rename_dict={'Tienda': 'Nombre_Sucursal'},        
    dropna_col='N° Accidentes',            
    post_rename={'Tienda': 'Nombre_Sucursal'},
    collect_for_merge=True 
)


Descuento_Productos_Dañados_2025 = procesar_base_sucursal(
    archivo_principal="APRE 2025.xlsx",
    sheet_name="Reporte_Ventas (37)",
    detalle_sucursales=Detalle_sucursales,    
    columnas_seleccion=['Código Sucursal', 'Descto'], 
    dropna_col='Descto',                       
    handle_productos_danados=True,
    post_rename={'Tienda': 'Nombre_Sucursal'},
    collect_for_merge=True
)

Merma_Destruccion_2025 = procesar_base_sucursal(
    archivo_principal="Detalle merma (18).xlsx",
    sheet_name="Export",
    detalle_sucursales=Detalle_sucursales,
    mapeo_outlets=mapeo_outlets,
    columnas_seleccion=['Sucursal', 'Costo_Merma', 'Unidades_Merma'],
    rename_dict={
        'Sucursal': 'ID_Sucursal_RAW',
        'Costo_Merma': 'Costo_Merma_Destruccion',
        'Unidades_Merma': 'Merma_Destruccion_Unidades'
    },
    dropna_col='Costo_Merma_Destruccion',
    handle_merma_destruccion=True,
    post_rename={'Costo_Merma_Destruccion': 'Costo_Merma_Destruccion'},
    dropna_all=True,
    collect_for_merge=True
)

Multas_Sucursal_2025 = procesar_base_sucursal(
    archivo_principal="202506 Informe de Juicios Retail - Consolidado 1.xlsx",
    detalle_sucursales=Detalle_sucursales,
    handle_multas=True,
    multas_sheet="JPL",
    multas_header=3,
    multas_nombre_col="Nombre_Sucursal",
    exclude_multas=["Marketplace", "Barros Arana"],
    post_rename={"Cantidad_Multas": "N° Multas"},
    collect_for_merge=True
)

Actividad_Criminal_2025 = procesar_base_sucursal(
    archivo_principal="Actividad Criminal.xlsx",
    detalle_sucursales=Detalle_sucursales,
    sheet_name="query",
    handle_actividad_criminal=True,
    actividad_nombre_cols=['Título'],
    manual_name_map=map_manual_default,
    post_rename={'Tienda': 'Nombre_Sucursal'},
    exclude_names={"Ofi Cerro Colorado 5240"},
    collect_for_merge=True,
)


Matriz_BHT = procesar_base_sucursal(
    archivo_principal="Matriz BHT.xlsx",
    dropna_all=True,
    post_rename={'Tienda': 'Nombre_Sucursal'},
    collect_for_merge=True

)

Metros_Cuadrados = procesar_base_sucursal(
    archivo_principal="Metros Cuadrados Sucursal.xlsx",
    columnas_seleccion=['Tienda', 'Mts. 2'],
    post_rename = {'Tienda': 'Nombre_Sucursal', 'Mts. 2': 'Mts. 2'},
    collect_for_merge=True
)

Rotacion_Personal_2025 = procesar_base_sucursal(
    archivo_principal= "Rotación por tienda 24-25.xlsx",
    handle_rotacion=True,
    rotacion_sheet="Rot. Total",
    exclude_names={"Alto Las Condes", "Huerfanos", "Concepción Barros Arana"},
    collect_for_merge=True,

)

Reclamos_Sucursal_2025 = procesar_base_sucursal(
    archivo_principal="Reclamos-Mala-Atencion-Tienda-2024-2025 (original).xlsx",
    handle_reclamos=True,
    reclamos_year_col='Año',          # por defecto 'Año'
    reclamos_ticket_col='Ticket',     # por defecto 'Ticket'
    reclamos_store_col='Tienda',      # por defecto 'Tienda'
    detalle_sucursales=Detalle_sucursales,
    manual_name_map=None,             # opcional: dict con correcciones específicas
    collect_for_merge=True,           # añadir resultado a MERGE_DFS para el merge maestro
    exclude_names = {"Altos Las Condes", "Concepcion 1"} ,               # opcional: set/list para excluir sucursales
    exclude_regex=None                # opcional: regex para excluir
)

Juicios_Division_2025_final = procesar_base_sucursal(
    archivo_principal="Consolidado Juicios Negocio completo 1 (1).xlsx",
    handle_juicios_laborales=True,
    juicios_sheet="2024",
    juicios_header=0,
    juicios_nombre_col="División de Personal",
    juicios_fecha_col="Fecha Audiencia",
    juicios_year=2025,
    collect_for_merge=True  
)


Df_Kmeans = merge_all_on_nombre_sucursal(MERGE_DFS, how='outer')
print(f"MASTER_SUCURSALES generado. Dimensiones: {Df_Kmeans.shape}")
# ========================= Post-proceso consolidado =========================
def preparar_consolidado_para_modelo(df):
    df = df.copy()

    # --- 1) Normalizaciones mínimas de nombres ---
    # unificar 'Juicios_Laborales' si vino con otro nombre
    cand_juicios = ['Juicios_Laborales', 'Juicios Laborales', 'Cantidad_Juicios_2025']
    jcol = next((c for c in cand_juicios if c in df.columns), None)
    if jcol and jcol != 'Juicios_Laborales':
        df['Juicios_Laborales'] = df[jcol]

    # detectar columna de multas con variantes
    cand_multas = ['N° Multas', 'Nº Multas', 'Nro Multas', 'Numero Multas']
    mcol = next((c for c in cand_multas if c in df.columns), None)

    # --- 2) Rellenos solicitados ---
    if mcol:
        df[mcol] = pd.to_numeric(df[mcol], errors='coerce').fillna(0)
    if 'Juicios_Laborales' in df.columns:
        df['Juicios_Laborales'] = pd.to_numeric(df['Juicios_Laborales'], errors='coerce').fillna(0)

    # --- 3) Redondeo de costo de merma ---
    if 'Costo_Merma_Destruccion' in df.columns:
        df['Costo_Merma_Destruccion'] = pd.to_numeric(df['Costo_Merma_Destruccion'], errors='coerce').round(2)

    # --- 4) Ventas / Mts. 2 ---
    if {'Total_Ventas', 'Mts. 2'}.issubset(df.columns):
        v = pd.to_numeric(df['Total_Ventas'], errors='coerce')
        m2 = pd.to_numeric(df['Mts. 2'], errors='coerce')
        df['Ventas / Mts. 2'] = np.divide(v, m2, out=np.full_like(v, np.nan, dtype=float), where=(m2 > 0))

    # --- 5) NC / Ventas (robusto + winsorización 99) ---
    if {'Total_Ventas', 'Total_Notas_Credito'}.issubset(df.columns):
        den = pd.to_numeric(df["Total_Ventas"], errors='coerce').astype(float)
        num = pd.to_numeric(df["Total_Notas_Credito"], errors='coerce').astype(float)

        ratio = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
        ratio[~np.isfinite(ratio)] = np.nan
        ratio[ratio < 0] = np.nan
        p99 = np.nanpercentile(ratio, 99) if np.isfinite(np.nanpercentile(ratio, 99)) else np.nan
        if np.isfinite(p99):
            ratio = np.clip(ratio, 0, p99)
        ratio = np.nan_to_num(ratio, nan=0.0)
        df["NC / Ventas"] = ratio

    # --- 6) Índice de Inseguridad = 1 - Indice Seguridad (0-1 o 0-100) ---
    if 'Indice Seguridad' in df.columns:
        seg = pd.to_numeric(df['Indice Seguridad'], errors='coerce')
        seg01 = seg / 100.0 if pd.to_numeric(seg, errors='coerce').max(skipna=True) > 1.5 else seg
        df['Indice_Inseguridad'] = (1.0 - seg01).clip(0, 1)

    return df

Df_Kmeans = merge_all_on_nombre_sucursal(MERGE_DFS, how='outer')
Df_Kmeans = preparar_consolidado_para_modelo(Df_Kmeans)  # <-- aplica reglas pedidas

try:
    out_path = os.path.join(OUTPUT_DIR, "Df_Kmeans.xlsx")
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        Df_Kmeans.to_excel(writer, index=False, sheet_name='Consolidado_Sucursales')
        autofit_columns(writer.sheets['Consolidado_Sucursales'], Df_Kmeans, index=False)
    print("Df_Kmeans guardado en:", out_path)
except NameError:
    print("Variable Df_Kmeans no existe. Asegúrate de ejecutar todas las celdas del notebook.")

def ejecutar_modelo_kmeans_minmax_pesos(
    df_consolidado,
    variables_modelo,
    pesos,
    k=3,
    random_state=100,
    n_init=50,
    outdir=OUTPUT_DIR,
    nombre_base="Propuesta_6"
):
    """
    - Escala MinMax las variables
    - Aplica ponderación geométrica con sqrt(peso) para KMeans
    - KMeans con k clusters
    - Calcula RiskScore (suma lineal de pesos sobre MinMax)
    - Etiqueta riesgos por promedio de RiskScore por cluster
    - Devuelve y guarda Reporte, Perfil, Top10, Centroides (en crudo y en original)
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # -------- 0) Chequeos básicos --------
    df = df_consolidado.copy()
    faltantes = [v for v in variables_modelo if v not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en Df_Kmeans para el modelo: {faltantes}")

    # Columna de sucursal para reportes (opcional)
    cand_cols = [c for c in df.columns if c.lower() in
                 ['nombre_sucursal','sucursal','tienda','cod_sucursal','código_sucursal','codigo_sucursal']]
    col_sucursal = cand_cols[0] if cand_cols else None

    # -------- 1) Subset y MinMax --------
    X_raw = df[variables_modelo].astype(float).copy()

    # Rellenos defensivos (si los hubiera)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    # Imputación simple sólo para el escalado; el consolidado ya trae NaN manejados en tus pasos
    X_raw = X_raw.fillna(0.0)

    MinMax_Scaler = MinMaxScaler()
    X_minmax = pd.DataFrame(
        MinMax_Scaler.fit_transform(X_raw),
        index=df.index,
        columns=variables_modelo
    )

    # -------- 2) Ponderación geométrica (sqrt(peso)) --------
    # Si algún peso no está en el dict, usa 0 por seguridad (no aporta a distancia)
    vector_pesos = np.array([pesos.get(v, 0.0) for v in variables_modelo], dtype=float)
    sqrt_pesos = np.sqrt(vector_pesos)
    X_weighted = X_minmax.values * sqrt_pesos

    # -------- 3) K-Means --------
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        init="k-means++",
        algorithm="lloyd"
    )
    clusters = kmeans.fit_predict(X_weighted)

    # -------- 4) Risk score (lineal con PESOS, no sqrt) --------
    Matriz_Risk = X_minmax.copy()
    missing_cols = [col for col in pesos.keys() if col not in Matriz_Risk.columns]
    for col, w in pesos.items():
        if col in Matriz_Risk.columns:
            Matriz_Risk[col] = Matriz_Risk[col] * w
    if missing_cols:
        print(f"(Nota) Se ignoraron pesos para columnas ausentes: {missing_cols}")

    risk_score = Matriz_Risk.sum(axis=1)

    # -------- 5) Etiquetado por riesgo (promedio por cluster) --------
    df_tmp = pd.DataFrame({"Cluster_id": clusters, "Risk": risk_score})
    risk_prom = df_tmp.groupby("Cluster_id")["Risk"].mean().sort_values()
    # rank (0..k-1) por riesgo promedio
    mapping_rank = {cid: rank for rank, cid in enumerate(risk_prom.index)}

    # nombres por riesgo (para 3 clusters)
    etiquetas_3 = {0:"Riesgo Bajo", 1:"Riesgo Medio", 2:"Riesgo Alto"}
    etiqueta_default = lambda r: etiquetas_3.get(r, f"Riesgo {r}")

    df["Cluster"] = clusters
    df["RiskScore"] = risk_score.values
    df["Cluster_Riesgo"] = df["Cluster"].map(mapping_rank).map(etiqueta_default)

    # -------- 6) Reporte (con variables crudas) --------
    cols_reporte = ([col_sucursal] if col_sucursal else [])
    reporte = pd.concat([
        df[cols_reporte + ["Cluster", "Cluster_Riesgo", "RiskScore"]],
        X_raw
    ], axis=1)

    # -------- 7) Perfil de clúster (promedio crudo) --------
    perfil = (reporte
              .groupby("Cluster_Riesgo")[variables_modelo]
              .mean())
    # Redondeo fino como definiste
    perfil = perfil.copy()
    for c in perfil.columns:
        perfil[c] = perfil[c].round(5 if c == "Desviacion_Meta_Merma" else 2)

    # -------- 8) Top 10 dentro del riesgo más alto disponible --------
    niveles = ["Riesgo Muy Alto", "Riesgo Alto", "Riesgo Medio", "Riesgo Bajo"]
    # Encuentra el más crítico presente
    nivel_top = next((n for n in niveles if n in df["Cluster_Riesgo"].values), None)
    if nivel_top is None:
        top10 = reporte.sort_values("RiskScore", ascending=False).head(10)
    else:
        mask = (df["Cluster_Riesgo"].values == nivel_top)
        top10 = (reporte.loc[mask]
                 .sort_values("RiskScore", ascending=False)
                 .head(10))

    # -------- 9) Centroides (weighted, minmax y original) --------
    # a) centroide en espacio ponderado (ya lo da kmeans.cluster_centers_)
    cent_weighted = pd.DataFrame(kmeans.cluster_centers_, columns=variables_modelo)
    # b) llevarlos a espacio minmax: dividir por sqrt_pesos
    with np.errstate(invalid='ignore', divide='ignore'):
        cent_minmax = cent_weighted.values / sqrt_pesos
    cent_minmax = pd.DataFrame(cent_minmax, columns=variables_modelo)
    # c) llevar a escala original
    cent_original = pd.DataFrame(
        MinMax_Scaler.inverse_transform(cent_minmax),
        columns=variables_modelo
    )

    try:
        reporte_ajustado = aplicar_ajustes_post_modelo(
            reporte,
            nombre_col_riesgo_base='Cluster_Riesgo',
            imprimir=False
        )
    except Exception as e:
        print(f"⚠ Error aplicando ajustes post-modelo: {e}")
        reporte_ajustado = reporte.copy()

    # Subconjuntos de Resultados # 
    riesgo_alto_df = reporte_ajustado.loc[reporte_ajustado['Riesgo_final'] == 'Riesgo Alto']
    riesgo_medio_df = reporte_ajustado.loc[reporte_ajustado['Riesgo_final'] == 'Riesgo Medio']

    # ===== Generamos Grafico ===== #
    png_path = None
    try: 
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        from matplotlib.patches import Polygon
        import matplotlib.pyplot as plt

        df_plot = reporte_ajustado.copy()
        store_col = next((c for c in ["Nombre_Sucursal", "sucursal", "cod_sucursal", "Tienda", "codigo_sucursal", "código_sucursal"] if c in df_plot.columns), None)
        risk_col = 'Riesgo_final' if 'Riesgo_final' in df_plot.columns else 'Cluster_Riesgo'
        niveles = ['Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto']
        color_map = {
            'Riesgo Bajo': '#2ca02c',    # verde
            'Riesgo Medio': "#ebe70d",  # amarillo
            'Riesgo Alto': '#d62728',   # rojo
        }
        if "RiskScore" in df_plot.columns:
            bubble_size = (pd.to_numeric(df_plot["RiskScore"], errors='coerce').fillna(0).values * 600 + 20)
        elif "RiskScore__P3" in df_plot.columns:
            bubble_size = (pd.to_numeric(df_plot["RiskScore__P3"], errors='coerce').fillna(0).values * 600 + 20)
        else:
            bubble_size = np.full(len(df_plot), 120.0)

        x = pd.to_numeric(df_plot["Ventas / Mts. 2"], errors='coerce').fillna(0).values

        risk_vars = [v for v in ["Productos_Dañados_Descuento", "Costo_Merma_Destruccion", "Desviacion_Meta_Merma",]
                     if v in X_minmax.columns]
        if len(risk_vars) >= 2:
            pc1 = PCA(n_components=1, random_state=100).fit_transform(X_minmax[risk_vars].values).ravel()
            if "RiskScore" in df_plot.columns and np.isfinite(pc1).any():
                corr = np.corrcoef(
                    pc1, 
                    pd.to_numeric(df_plot["RiskScore"], errors='coerce').fillna(0).values
                )[0, 1]
                if corr < 0:
                    pc1 = -pc1
            y = (pc1 - np.nanmin(pc1)) / (np.nanmax(pc1) - np.nanmin(pc1) + 1e-12)
        else: 
            y_raw = pd.to_numeric(df_plot.get("Desviacion_Meta_Merma", 0), errors='coerce').fillna(0).values
            y = (y_raw - np.nanmin(y_raw)) / (np.nanmax(y_raw) - np.nanmin(y_raw) + 1e-12)

        def convex_hull_xy(xv, yv):
            pts = np.column_stack((xv, yv)).astype(float)
            pts = pts[~np.isnan(pts).any(axis=1)]  
            if len(pts) < 3: return None
            pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
            def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            lower = []
            for p in pts:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
                lower.append(tuple(p))
            upper = []
            for p in pts[::-1]:
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
                upper.append(tuple(p))
            return np.array(lower[:-1] + upper[:-1])
        
        def fmt_pesos_m2(val, pos):
            if val >= 1e6: return f"{val/1e6:,.1f} MM $/m²"
            if val >= 1e3: return f"{val/1e3:,.0f} mil $/m²"
            return f"{val:,.0f} $/m²"
        fig, ax = plt.subplots(figsize=(23, 18))
        labels_riesgo = df_plot[risk_col].astype(str).values

        for lvl in niveles:
            m = (labels_riesgo == lvl)
            if np.any(m):
                ax.scatter(x[m], y[m],
                            s=bubble_size[m],
                            c=color_map.get(lvl),
                            alpha=0.7,
                            edgecolors='none',
                            label=lvl,
                            zorder=2)
                if m.sum() >= 3:
                    poly = convex_hull_xy(x[m], y[m])
                    if poly is not None:
                        ax.add_patch(Polygon(poly , closed = True,
                                            facecolor=color_map[lvl],
                                            edgecolor=color_map[lvl],
                                            alpha=0.12, lw = 2, zorder=1))
        if store_col:
            for xi, yi, txt in zip(x, y, df_plot[store_col].astype(str).values):
                if np.isfinite(xi) and np.isfinite(yi):
                    ax.annotate(f" {txt}", (xi, yi), xytext = (2,2),
                    textcoords = "offset points" ,fontsize=8)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6)) 
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_pesos_m2))
        ax.set_xlabel("Ventas / Mts. 2($/m2)")
        ax.set_ylabel("Componente Principal (Inventario - Merma - Dañados)")
        ax.grid(True, linewidth = 0.3, alpha=0.45, zorder = 0)
        ax.legend(title="Nivel de Riesgo", frameon = True, loc = "upper left")
        plt.tight_layout()

        png_path = os.path.join(outdir, f"{nombre_base}_Grafico_Clustering.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"⚠ Error generando gráfico: {e}")
        png_path = None

         

    # -------- 10) Guardado a Excel --------
    os.makedirs(outdir, exist_ok=True)
    path_xlsx = os.path.join(outdir, f"{nombre_base}_Modelo.xlsx")

    reporte_ajustado = reporte_ajustado.copy()
    reporte_ajustado['Conclusiones'] = construir_conclusiones_ripley(
    reporte_base=reporte,           
    X_minmax=X_minmax,              
    pesos_dict=pesos,               
    reporte_final=reporte_ajustado, 
    top_n=3                          
)
    with pd.ExcelWriter(path_xlsx, engine="xlsxwriter") as writer:
        reporte_ajustado.to_excel(writer, sheet_name="Reporte", index=False)
        reporte_ajustado = reporte_ajustado.copy()
        reporte_ajustado['Conclusiones'] = construir_conclusiones_ripley(
            reporte_base=reporte,             # DF con valores crudos
            X_minmax=X_minmax,                # matriz min-max ya calculada en la función
            pesos_dict=pesos,                 # dict de pesos que usas en tu score
            reporte_final=reporte_ajustado,   # contiene Riesgo_final + NC/Ventas + Motivo_Regla
            top_n=3                           # se ignora si Riesgo=Alto (usa 4)
            )
        perfil.to_excel(writer, sheet_name="Perfil_Cluster", index=True)
        riesgo_alto_df.to_excel(writer, sheet_name="Riesgo Alto", index=False)
        riesgo_medio_df.to_excel(writer, sheet_name="Riesgo Medio", index=False)
    

        if png_path and os.path.exists(png_path):
            ws_chart = writer.book.add_worksheet("Gráfico_Clustering")
            ws_chart.insert_image('A1', png_path, {'x_scale': 0.85, 'y_scale': 0.85})
        
        try:
            map_html = os.path.join(outdir, "mapa_ripley.html")
            df_map, presentes, colores = build_map_df(reporte_ajustado)
            render_map_and_save(df_map, presentes, colores, map_html)
            add_map_sheet_xlsxwriter(writer, map_html)  
        except Exception as e:
            print(f"⚠ No se pudo generar hoja 'Mapa Interactivo': {e}")

        sheets_to_autofit = [
            ("Reporte", reporte_ajustado, False),
            ("Perfil_Cluster", perfil, True),
            ("Riesgo Alto", riesgo_alto_df, False),
            ("Riesgo Medio", riesgo_medio_df, False),
        ]
        for sheet_name, df_, use_index in sheets_to_autofit:
            ws = writer.sheets.get(sheet_name)
            if ws is not None:
                autofit_columns(ws, df_, index=use_index)

    print(f"[OK] Modelo guardado en: {path_xlsx}")

    return {
        "reporte": reporte_ajustado,
        "perfil": perfil,
        "Riesgo Alto": riesgo_alto_df,
        "kmeans": kmeans,
        "scaler": MinMax_Scaler,
        "centroides_original": cent_original,
    }
# ================== AJUSTES POST-MODELO (mínimos) ==================
def aplicar_ajustes_post_modelo(
    reporte,
    nombre_col_riesgo_base='Cluster_Riesgo',
    ventas_min=1e6,          
    abs_thr=0.04,            
    perc_q=0.90,             
    imprimir=True
):
    """
    - Crea/actualiza 'Riesgo_final' partiendo de 'Cluster_Riesgo'
    - Ajuste 1: si 'Productos_Dañados_Descuento' >= p85 y era 'Riesgo Bajo' => 'Riesgo Medio'
    - Ajuste 2: si 'NC / Ventas' >= max(4%, p90) con ventas >= 1M => 'Riesgo Medio'
    - Devuelve un nuevo DataFrame 'reporte' con 'Riesgo_final' y 'NC / Ventas'
    """
    rep = reporte.copy()

    # 0) columna base de riesgo
    if nombre_col_riesgo_base not in rep.columns:
        raise KeyError(f"No encuentro la columna base de riesgo: {nombre_col_riesgo_base}")
    rep['Riesgo_final'] = rep[nombre_col_riesgo_base].astype(str)

    # 1) Ajuste por Productos_Dañados_Descuento (p85)
    if 'Productos_Dañados_Descuento' in rep.columns:
        p85 = pd.to_numeric(rep['Productos_Dañados_Descuento'], errors='coerce').quantile(0.85)
        mask_dan = (
            (rep['Riesgo_final'] == 'Riesgo Bajo') &
            (pd.to_numeric(rep['Productos_Dañados_Descuento'], errors='coerce') >= p85)
        )
        rep.loc[mask_dan, 'Riesgo_final'] = 'Riesgo Medio'

    # 2) Ajuste por NC / Ventas (>= max(4%, p90)) con ventas >= 1M
    if {'Total_Ventas','Total_Notas_Credito'}.issubset(rep.columns):
        vt = pd.to_numeric(rep['Total_Ventas'], errors='coerce')
        nc = pd.to_numeric(rep['Total_Notas_Credito'], errors='coerce')
        ratio = nc / vt.replace(0, np.nan)
        ratio[vt < ventas_min] = np.nan  # evita ruido por ventas chicas
        p90 = ratio.dropna().quantile(perc_q) if ratio.notna().any() else abs_thr
        thr = max(abs_thr, float(p90) if pd.notna(p90) else abs_thr)
        rep['NC / Ventas'] = ratio
        mask_nc = (rep['Riesgo_final'] == 'Riesgo Bajo') & (rep['NC / Ventas'] >= thr)
        rep.loc[mask_nc, 'Riesgo_final'] = 'Riesgo Medio'
        if imprimir:
            print(f"Umbral NC/Ventas usado: {thr:.2%} (max(4%, p90={p90:.2%})).")
    else:
        rep['NC / Ventas'] = np.nan

    # Orden categórico estándar
    orden = pd.CategoricalDtype(['Riesgo Bajo','Riesgo Medio','Riesgo Alto','Riesgo Muy Alto'], ordered=True)
    rep['Riesgo_final'] = rep['Riesgo_final'].astype(orden)

    return rep
# ================== FIN AJUSTES POST-MODELO ==================

# ================== Ejecutar modelo (k=3, decisión corporativa) ==================
Variables_Modelo = [
    "Ventas / Mts. 2", "Total_Notas_Credito",
    "Reclamos_Cantidad_2025", "Desviacion_Meta_Merma",
    "Costo_Merma_Destruccion", "N° Accidentes",
    "Productos_Dañados_Descuento", "N° Multas",
    "N° Recuperos 2025", "Indice_Inseguridad",
    "Rot_2025", "Juicios_Laborales",
    "Total_Ventas"
]

Weights = {
    'Ventas / Mts. 2': 0.13,
    'Total_Notas_Credito': 0.100,
    'Desviacion_Meta_Merma': 0.155,
    'Productos_Dañados_Descuento': 0.145,
    'Costo_Merma_Destruccion': 0.14,
    'N° Accidentes': 0.070,
    'Reclamos_Cantidad_2025': 0.070,
    "N° Multas": 0.03,
    "N° Recuperos 2025": 0.050,
    "Indice_Inseguridad": 0.010,
    "Rot_2025": 0.03,
    "Juicios_Laborales": 0.03, 
    "Total_Ventas": 0.0
}

res_modelo = ejecutar_modelo_kmeans_minmax_pesos(
    df_consolidado=Df_Kmeans,
    variables_modelo=Variables_Modelo,
    pesos=Weights,
    k=3,
    random_state=100,
    n_init=50,
    outdir=OUTPUT_DIR,
    nombre_base="Resultados_Clustering"
)
# === Ajustes Post - Modelo sobre el Reporte ===
reporte_ajustado = aplicar_ajustes_post_modelo(res_modelo["reporte"], imprimir=True)
# Si quieres que Df_Kmeans conserve las 3 columnas claves del modelo:
Df_Kmeans = Df_Kmeans.copy()
Df_Kmeans["Cluster"] = res_modelo["reporte"]["Cluster"].values
Df_Kmeans["Cluster_Riesgo"] = res_modelo["reporte"]["Cluster_Riesgo"].values
Df_Kmeans["RiskScore"] = res_modelo["reporte"]["RiskScore"].values
Df_Kmeans["Riesgo_final"] = reporte_ajustado["Riesgo_final"].astype(str).values
Df_Kmeans["NC / Ventas"]  = reporte_ajustado["NC / Ventas"].values

# Guardar el consolidado final con columnas del modelo
out_master = os.path.join(OUTPUT_DIR, "Df_Kmeans.xlsx")
with pd.ExcelWriter(out_master, engine="xlsxwriter") as writer:
    Df_Kmeans.to_excel(writer, sheet_name="Df_Kmeans", index=False)
    autofit_columns(writer.sheets["Df_Kmeans"], Df_Kmeans, index=False)
    print("Df_Kmeans actualizado con columnas del modelo en:", out_master)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline consolidado + Modelo k-means con pesos")
    parser.add_argument("--input", default=INPUT_DIR, help="Carpeta input (con archivos fuente)")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Carpeta output para resultados")
    parser.add_argument("--k", type=int, default=3, help="Número de clusters (default=3)")
    parser.add_argument("--seed", type=int, default=100, help="Random state (default=100)")
    args = parser.parse_args()

    # Reajustar rutas si vienen por CLI
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === Aquí corre TODO tu flujo actual de carga/merge/preparación ===
    # (No lo repito para no duplicar; mantén exactamente tus llamadas existentes)
    # ...
    # Df_Kmeans = merge_all_on_nombre_sucursal(MERGE_DFS, how='outer')
    # Df_Kmeans = preparar_consolidado_para_modelo(Df_Kmeans)
    # ...

    # === Ejecutar el modelo con k de CLI ===
    res_modelo = ejecutar_modelo_kmeans_minmax_pesos(
        df_consolidado=Df_Kmeans,
        variables_modelo=Variables_Modelo,
        pesos=Weights,
        k=args.k,
        random_state=args.seed,
        n_init=50,
        outdir=OUTPUT_DIR,
        nombre_base="Resultados_Clustering"
    )
    # === AJUSTES POST-MODELO: aplica sobre el reporte devuelto por el modelo ===
    reporte_ajustado = aplicar_ajustes_post_modelo(res_modelo["reporte"])
    # Propagar a tu consolidado sin tocar el resto
    Df_Kmeans["Riesgo_final"] = reporte_ajustado["Riesgo_final"].astype(str).values
    Df_Kmeans["NC / Ventas"]  = reporte_ajustado["NC / Ventas"].values

    # Persistir consolidado final
    out_master = os.path.join(OUTPUT_DIR, "Df_Kmeans.xlsx")
    with pd.ExcelWriter(out_master, engine="xlsxwriter") as writer:
        Df_Kmeans.to_excel(writer, sheet_name="Df_Kmeans", index=False)
        autofit_columns(writer.sheets["Df_Kmeans"], Df_Kmeans, index=False)
    Df_Kmeans["Cluster"] = res_modelo["reporte"]["Cluster"].values
    Df_Kmeans["Cluster_Riesgo"] = res_modelo["reporte"]["Cluster_Riesgo"].values
    Df_Kmeans["RiskScore"] = res_modelo["reporte"]["RiskScore"].values
    print("Df_Kmeans actualizado con columnas del modelo en:", out_master)
# === END: Code extracted from notebook ===
