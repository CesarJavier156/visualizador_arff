import matplotlib
matplotlib.use('Agg')  # Usa backend sin interfaz gráfica
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
import arff
from django.shortcuts import render

def upload_file(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            file = request.FILES['file']
            dataset = arff.load(file)
            df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

            # === Graficar solo 4 columnas si hay muchas ===
            sample_cols = df.columns[:4] if len(df.columns) >= 4 else df.columns

            images = []
            for col in sample_cols:
                plt.figure(figsize=(5, 3))
                sns.histplot(df[col], kde=True)
                plt.title(f"Distribución de {col}")
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                images.append(image_base64)
                plt.close()  # 🔥 Libera memoria después de cada gráfica

            context['columns'] = df.head().to_html(classes='table table-dark table-striped')
            context['images'] = images

        except Exception as e:
            context['error'] = f"Ocurrió un error procesando el archivo: {e}"

    return render(request, 'index.html', context)
