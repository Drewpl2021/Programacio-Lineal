from flask import Flask, render_template, request
import numpy as np
from scipy.optimize import linprog

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

    ax.plot(res.x[0], res.x[1], 'ro', markersize=12, label='Máximo' if es_max else 'Mínimo')
    signo = "Máximo" if es_max else "Mínimo"
    ax.annotate(f'{signo} G={abs(res.fun):.2f}\n({res.x[0]:.2f}, {res.x[1]:.2f})',
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

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    grafica = None
    error = None
    es_max = True  # para mostrar texto correcto

    if request.method == 'POST':
        try:
            # Leer función objetivo y todas las restricciones dinámicas
            f_obj_text = request.form['funcion_objetivo']
            restricciones = request.form.getlist('restricciones')  # todas las restricciones

            maximizar = 'maximizar' in request.form  # True si checkbox está marcado

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
                c = -c  # maximizar con linprog que minimiza

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

                vertices = []
                n = len(A_ub)
                for i in range(n):
                    for j in range(i+1, n):
                        det = A_ub[i][0]*A_ub[j][1] - A_ub[j][0]*A_ub[i][1]
                        if abs(det) < 1e-10:
                            continue
                        xv = (b_ub[i]*A_ub[j][1] - b_ub[j]*A_ub[i][1])/det
                        yv = (A_ub[i][0]*b_ub[j] - A_ub[j][0]*b_ub[i])/det
                        p = np.array([xv,yv])
                        if np.all(np.dot(A_ub, p) <= b_ub + 1e-5) and np.all(p >= -1e-5):
                            vertices.append(p)
                vertices = np.array(vertices)

                grafica = graficar_2d(A_ub, b_ub, vertices, res, es_max)

        except Exception as e:
            error = str(e)

    return render_template('index.html', resultado=resultado, grafica=grafica, error=error, es_max=es_max)

if __name__ == '__main__':
    app.run(debug=True)
