from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.io import to_html

import matplotlib
matplotlib.use('Agg')  # <<-- Esta línea debe ir antes de importar pyplot

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import io
import base64

app = Flask(__name__)

def parse_func_objetivo_2d(texto):
    texto = texto.replace(' ', '')  # elimina espacios
    texto = texto.replace('-', '+-')
    partes = texto.split('+')
    coef = {'x': 0, 'y': 0}
    for parte in partes:
        parte = parte.strip()
        if parte == '':
            continue
        if 'x' in parte:
            num = parte.replace('x', '') or '1'
            coef['x'] = float(num)
        elif 'y' in parte:
            num = parte.replace('y', '') or '1'
            coef['y'] = float(num)
    return [coef['x'], coef['y']]

def parse_restriccion_2d(texto):
    texto = texto.replace(' ', '')  # elimina espacios
    texto = texto.replace('-', '+-')
    if '<=' in texto:
        izq, der = texto.split('<=')
        tipo = '<='
    elif '>=' in texto:
        izq, der = texto.split('>=')
        tipo = '>='
    elif '=' in texto:
        izq, der = texto.split('=')
        tipo = '='
    else:
        raise ValueError("Restricción debe contener <=, >= o =")

    coef = [0, 0]

    # parsear términos sin romper los signos negativos
    partes = []
    current = ''
    for c in izq:
        if c == '+' and current != '':
            partes.append(current)
            current = ''
        else:
            current += c
    if current != '':
        partes.append(current)

    for parte in partes:
        parte = parte.strip()
        if parte == '':
            continue
        if 'x' in parte:
            num = parte.replace('x', '') or '1'
            coef[0] = float(num)
        elif 'y' in parte:
            num = parte.replace('y', '') or '1'
            coef[1] = float(num)

    return coef, float(der.strip()), tipo

def graficar_2d(A, b, vertices, res, es_max):
    fig = go.Figure()

    # Definir rango de valores en X
    x_vals = np.linspace(0, max(vertices[:, 0]) * 1.5, 400)

    # Dibujar restricciones
    for i, (coef, val) in enumerate(zip(A, b)):
        a, c = coef
        if abs(c) > 1e-10:
            y_vals = (val - a * x_vals) / c
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f"Restricción {i + 1}",
                                     line=dict(dash='dash')))
        else:
            x_line = val / a
            fig.add_shape(type="line", x0=x_line, x1=x_line, y0=0, y1=max(vertices[:, 1]) * 1.5,
                          line=dict(dash='dash'), name=f"Restricción {i + 1}")

    # Región factible
    fig.add_trace(go.Scatter(x=vertices[:, 0], y=vertices[:, 1], fill="toself", name="Región Factible",
                             mode="lines", fillcolor='lightgreen', line=dict(color='green')))

    # Vértices
    fig.add_trace(go.Scatter(x=vertices[:, 0], y=vertices[:, 1], mode='markers+text', name="Vértices",
                             marker=dict(color='blue', size=10),
                             text=[f"V{i + 1}" for i in range(len(vertices))], textposition="top center"))

    # Solución óptima
    fig.add_trace(go.Scatter(x=[res.x[0]], y=[res.x[1]], mode='markers+text', name="Óptimo",
                             marker=dict(color='red', size=12),
                             text=[f"{'Máximo' if es_max else 'Mínimo'}<br>G={abs(res.fun):.2f}"],
                             textposition="bottom right"))

    # 👉 Marcar el origen (0,0)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', name="Origen (0,0)",
                             marker=dict(color='black', size=8, symbol='x'),
                             text=["(0,0)"], textposition="bottom right"))

    # Ajustar layout del gráfico
    fig.update_layout(
        title="Gráfica Interactiva 2D",
        xaxis=dict(
            title="x",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            showgrid=True
        ),
        yaxis=dict(
            title="y",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            showgrid=True
        ),
        width=800,
        height=600,
        showlegend=True
    )

    return to_html(fig, full_html=False)



@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    grafica_html = None
    error = None
    es_max = True  # para mostrar texto correcto

    if request.method == 'POST':
        try:
            f_obj_text = request.form['funcion_objetivo']
            restricciones = request.form.getlist('restricciones')
            maximizar = 'maximizar' in request.form

            f_obj = parse_func_objetivo_2d(f_obj_text)

            A_ub = []
            b_ub = []

            for r in restricciones:
                coef, val, tipo = parse_restriccion_2d(r)
                if tipo == '<=':
                    A_ub.append(coef)
                    b_ub.append(val)
                elif tipo == '>=':
                    A_ub.append([-c for c in coef])
                    b_ub.append(-val)
                elif tipo == '=':
                    A_ub.append(coef)
                    b_ub.append(val)
                    A_ub.append([-c for c in coef])
                    b_ub.append(-val)

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            c = np.array(f_obj)
            if maximizar:
                c = -c  # linprog solo minimiza

            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, method='highs')

            if not res.success:
                error = "No se encontró solución óptima."
            else:
                es_max = maximizar
                resultado = {
                    'valor_objetivo': -res.fun if es_max else res.fun,
                    'x': res.x[0],
                    'y': res.x[1]
                }

                # Calcular vértices factibles
                vertices = []
                n = len(A_ub)
                for i in range(n):
                    for j in range(i + 1, n):
                        det = A_ub[i][0] * A_ub[j][1] - A_ub[j][0] * A_ub[i][1]
                        if abs(det) < 1e-10:
                            continue
                        xv = (b_ub[i] * A_ub[j][1] - b_ub[j] * A_ub[i][1]) / det
                        yv = (A_ub[i][0] * b_ub[j] - A_ub[j][0] * b_ub[i]) / det
                        punto = np.array([xv, yv])
                        if np.all(np.dot(A_ub, punto) <= b_ub + 1e-5) and np.all(punto >= -1e-5):
                            vertices.append(punto)
                vertices = np.array(vertices)

                grafica_html = graficar_2d(A_ub, b_ub, vertices, res, es_max)

        except Exception as e:
            error = str(e)

    return render_template('index.html', resultado=resultado, grafica=grafica_html, error=error, es_max=es_max)


if __name__ == '__main__':
    app.run(debug=True)