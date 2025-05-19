from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import io
import base64
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# ---- Funciones comunes: parseo 2D y 3D ----

def parse_func_objetivo_2d(texto):
    texto = texto.replace('-', '+-')
    partes = texto.split('+')
    coef = {'x': 0, 'y': 0}
    for parte in partes:
        parte = parte.strip()
        if 'x' in parte:
            num = parte.replace('x', '') or '1'
            coef['x'] = float(num)
        elif 'y' in parte:
            num = parte.replace('y', '') or '1'
            coef['y'] = float(num)
    return [coef['x'], coef['y']]

def parse_restriccion_2d(texto):
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
    partes = izq.split('+')
    for parte in partes:
        parte = parte.strip()
        if 'x' in parte:
            num = parte.replace('x', '') or '1'
            coef[0] = float(num)
        elif 'y' in parte:
            num = parte.replace('y', '') or '1'
            coef[1] = float(num)
    return coef, float(der.strip()), tipo

def graficar_2d(A, b, vertices, res):
    fig, ax = plt.subplots(figsize=(8,6))
    x_vals = np.linspace(0, max(vertices[:,0])*1.5, 400)
    colores = ['#1f77b4', '#ff7f0e']

    for i, (coef, val) in enumerate(zip(A, b)):
        a, c = coef[0], coef[1]
        color = colores[i % len(colores)]
        if abs(c) > 1e-10:
            y_vals = (val - a*x_vals) / c
            y_vals = np.clip(y_vals, 0, 1e9)
            ax.plot(x_vals, y_vals, label=f"Restricción {i+1}", color=color, linewidth=2, linestyle='--')
        else:
            x_line = val / a
            ax.axvline(x=x_line, label=f"Restricción {i+1}", color=color, linewidth=2, linestyle='--')

    poligono = Polygon(vertices, closed=True, fill=True, edgecolor='green', facecolor='lightgreen', alpha=0.4, linewidth=2)
    ax.add_patch(poligono)

    ax.scatter(vertices[:,0], vertices[:,1], color='blue', s=80, label='Vértices')
    for i, (xv, yv) in enumerate(vertices):
        ax.text(xv, yv, f'V{i+1}', fontsize=12, ha='right', va='bottom', fontweight='bold')

    ax.plot(res.x[0], res.x[1], 'ro', markersize=12, label='Máximo')
    ax.annotate(f'Máximo G={-res.fun:.2f}\n({res.x[0]:.2f}, {res.x[1]:.2f})',
                (res.x[0], res.x[1]), textcoords='offset points', xytext=(15,-30), ha='left', fontsize=12, color='red', fontweight='bold')

    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_title('Región factible y solución óptima', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('ascii')
    return img_base64

# --- Funciones para 3D con plotly ---

def parse_func_objetivo_3d(texto):
    texto = texto.replace('-', '+-')
    partes = texto.split('+')
    coef = {'x': 0, 'y': 0, 'z': 0}
    for parte in partes:
        parte = parte.strip()
        if 'x' in parte:
            num = parte.replace('x', '') or '1'
            coef['x'] = float(num)
        elif 'y' in parte:
            num = parte.replace('y', '') or '1'
            coef['y'] = float(num)
        elif 'z' in parte:
            num = parte.replace('z', '') or '1'
            coef['z'] = float(num)
    return [coef['x'], coef['y'], coef['z']]

def parse_restriccion_3d(texto):
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

    coef = [0, 0, 0]
    partes = izq.split('+')
    for parte in partes:
        parte = parte.strip()
        if 'x' in parte:
            num = parte.replace('x', '') or '1'
            coef[0] = float(num)
        elif 'y' in parte:
            num = parte.replace('y', '') or '1'
            coef[1] = float(num)
        elif 'z' in parte:
            num = parte.replace('z', '') or '1'
            coef[2] = float(num)
    return coef, float(der.strip()), tipo

def graficar_3d_plotly(A, b, res):
    x = np.linspace(0, max(res.x[0]*2, 10), 30)
    y = np.linspace(0, max(res.x[1]*2, 10), 30)
    X, Y = np.meshgrid(x, y)

    surfaces = []
    colores = ['red', 'green', 'blue']

    for i, (coef, val) in enumerate(zip(A, b)):
        a, b_, c_ = coef
        if abs(c_) < 1e-6:
            # Plano vertical o no definido en z: lo omitimos
            continue
        Z = (val - a*X - b_*Y) / c_

        surfaces.append(go.Surface(z=Z, x=X, y=Y, opacity=0.4,
                                   colorscale=[[0, colores[i % 3]], [1, colores[i % 3]]],
                                   showscale=False,
                                   name=f'Restricción {i+1}'))

    punto = res.x
    max_val = -res.fun
    punto_trace = go.Scatter3d(x=[punto[0]], y=[punto[1]], z=[punto[2]],
                              mode='markers+text',
                              marker=dict(size=7, color='black'),
                              text=[f'Máximo<br>G={max_val:.2f}'],
                              textposition='top center',
                              name='Punto máximo')

    fig = go.Figure(data=surfaces + [punto_trace])
    fig.update_layout(scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'),
        title='Región factible 3D y punto máximo',
        width=800, height=700
    )
    # Convertir figura a HTML para enviar a plantilla
    html_plot = pio.to_html(fig, full_html=False)
    return html_plot

# --- Ruta principal ---

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    grafica_2d = None
    grafica_3d = None
    error = None

    modo = request.form.get('modo', '2D')  # Valor por defecto 2D

    try:
        if request.method == 'POST':
            if modo == '2D':
                f_obj_text = request.form['funcion_objetivo_2d']
                r1 = request.form['restriccion1_2d']
                r2 = request.form['restriccion2_2d']

                f_obj = parse_func_objetivo_2d(f_obj_text)

                A_ub, b_ub = [], []

                for r in [r1, r2]:
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

                res = linprog(c=[-c for c in f_obj], A_ub=A_ub, b_ub=b_ub, method='highs')

                if not res.success:
                    error = "No se encontró solución óptima."
                else:
                    vertices = []
                    n = len(A_ub)
                    for i in range(n):
                        for j in range(i+1, n):
                            det = A_ub[i][0]*A_ub[j][1] - A_ub[j][0]*A_ub[i][1]
                            if abs(det) < 1e-10:
                                continue
                            x = (b_ub[i]*A_ub[j][1] - b_ub[j]*A_ub[i][1])/det
                            y = (A_ub[i][0]*b_ub[j] - A_ub[j][0]*b_ub[i])/det
                            p = np.array([x,y])
                            if np.all(np.dot(A_ub, p) <= b_ub + 1e-5) and np.all(p >= -1e-5):
                                vertices.append(p)
                    vertices = np.array(vertices)

                    grafica_2d = graficar_2d(A_ub, b_ub, vertices, res)
                    resultado = {
                        'maximo': -res.fun,
                        'x': res.x[0],
                        'y': res.x[1]
                    }

            elif modo == '3D':
                f_obj_text = request.form['funcion_objetivo_3d']
                r1 = request.form['restriccion1_3d']
                r2 = request.form['restriccion2_3d']
                r3 = request.form['restriccion3_3d']

                restricciones = [r1, r2, r3]

                f_obj = parse_func_objetivo_3d(f_obj_text)

                A_ub, b_ub = [], []

                for r in restricciones:
                    coef, val, tipo = parse_restriccion_3d(r)
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

                res = linprog(c=[-c for c in f_obj], A_ub=A_ub, b_ub=b_ub, method='highs')

                if not res.success:
                    error = "No se encontró solución óptima."
                else:
                    resultado = {
                        'maximo': -res.fun,
                        'x': res.x[0],
                        'y': res.x[1],
                        'z': res.x[2]
                    }
                    grafica_3d = graficar_3d_plotly(A_ub, b_ub, res)

    except Exception as e:
        error = str(e)

    return render_template('index.html',
                           resultado=resultado,
                           grafica_2d=grafica_2d,
                           grafica_3d=grafica_3d,
                           error=error,
                           modo=modo)

if __name__ == '__main__':
    app.run(debug=True)
