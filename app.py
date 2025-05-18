import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import curve_fit

# Modelo exponencial
def exponencial(t, r, P0, t0):
    return P0 * np.exp(r * (t - t0))

# Modelo logístico
def logistica(t, K, P0, r, t0):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * (t - t0)))

st.title("Calculadora de Crecimiento Poblacional")

st.markdown("""
Esta calculadora permite ajustar modelos de crecimiento poblacional:
- *Exponencial*: crecimiento ilimitado.
- *Logístico*: crecimiento limitado por capacidad máxima.

Los modelos se resuelven también con la transformada de Laplace cuando es posible.
""")

# Elección del modelo
tipo_modelo = st.radio("Seleccione el modelo a usar:", ["Exponencial", "Logístico"])

# Número de datos
num_datos = st.number_input("Número de datos a ingresar:", min_value=2, max_value=20, value=5)

# Entrada de datos
data_cols = st.columns(2)
anios = []
poblaciones = []
for i in range(num_datos):
    with data_cols[0]:
        anio = st.number_input(f"Año {i+1}", value=2019 + i, key=f"anio_{i}")
        anios.append(anio)
    with data_cols[1]:
        poblacion = st.number_input(f"Población {i+1} (millones)", value=round(37.59 + i * 0.6, 2), format="%.2f", key=f"pop_{i}")
        poblaciones.append(poblacion)

# Año a predecir
pred_anio = st.number_input("Año para predecir la población", value=2024)

if st.button("Calcular modelo y predecir"):
    t = np.array(anios, dtype=float)
    P = np.array(poblaciones, dtype=float)

    if len(t) < 2:
        st.error("Se necesitan al menos dos datos.")
        st.stop()

    orden = np.argsort(t)
    t = t[orden]
    P = P[orden]

    datos_ordenados_texto = "\n".join([f"Año: {int(a)}, Población: {float(b):.2f} millones" for a, b in zip(t, P)])
    st.markdown("*Datos ordenados:*\n" + datos_ordenados_texto)

    try:
        t0 = t[0]

        if tipo_modelo == "Exponencial":
            lnP = np.log(P)
            coef = np.polyfit(t - t0, lnP, 1)  
            r = coef[0]
            lnP0 = coef[1]
            P0 = np.exp(lnP0)

            st.latex(f"P(t) = {P0:.4f} \\, e^{{{r:.4f}(t - {t0})}}")

            # Transformada de Laplace
            t_sym, s = sp.symbols('t s')
            P0_sym, r_sym = sp.symbols('P0 r')
            Y = sp.Function('Y')(s)
            lap_eq = sp.Eq(s * Y - P0_sym, r_sym * Y)
            Ysol = sp.solve(lap_eq, Y)[0]
            P_of_t = sp.inverse_laplace_transform(Ysol.subs({P0_sym: P0, r_sym: r}), s, t_sym)

            st.markdown("*Transformada de Laplace:*")
            st.latex(sp.latex(lap_eq))
            st.latex("Y(s) = " + sp.latex(Ysol))
            st.latex("P(t) = " + sp.latex(P_of_t))

            P_pred = exponencial(pred_anio, r, P0, t0)
            t_range = np.linspace(min(t), pred_anio, 200)
            P_model = exponencial(t_range, r, P0, t0)

        else:
            def logistic_func(t, K, r):
                return logistica(t, K, P[0], r, t0)

            popt, _ = curve_fit(lambda t, K, r: logistic_func(t, K, r), t, P, p0=[max(P)*1.2, 0.1], bounds=(0, [1e9, 1.0]))
            K, r = popt
            P0 = P[0]

            st.latex(f"P(t) = {K:.2f} / (1 + (({K - P0:.2f}) / {P0:.2f}) e^{{-{r:.4f}(t - {t0})}})")

            # Solución simbólica
            t_sym, s = sp.symbols('t s')
            K_sym, P0_sym, r_sym = sp.symbols('K P0 r', positive=True, real=True)
            P_sol = K_sym / (1 + ((K_sym - P0_sym) / P0_sym) * sp.exp(-r_sym * (t_sym - t0)))
            st.markdown("*Transformada de Laplace (solución completa con Bernoulli):*")
            st.latex("P(t) = " + sp.latex(P_sol.subs({K_sym: K, P0_sym: P0, r_sym: r})))

            P_pred = logistica(pred_anio, K, P0, r, t0)
            t_range = np.linspace(min(t), pred_anio, 200)
            P_model = logistica(t_range, K, P0, r, t0)

        # Mostrar resultado
        st.success(f"Población estimada en {pred_anio}: {P_pred:.2f} millones")

        if P_pred < P[-1]:
            st.warning("⚠ La población predicha es menor a la del último año. Revisa los datos o el modelo.")

        # Comparación si existe
        if pred_anio in t:
            real_val = P[np.where(t == pred_anio)[0][0]]
            error_pct = abs((real_val - P_pred) / real_val) * 100
            st.warning(f"Porcentaje de error: {error_pct:.4f}%")

    except Exception as e:
        st.error(f"Error al calcular el modelo: {e}")
