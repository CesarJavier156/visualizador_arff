import io
import base64
import matplotlib.pyplot as plt
from django.shortcuts import render
from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split

def index(request):
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            # Leer archivo ARFF
            arff_file = request.FILES['file']
            data, meta = arff.loadarff(arff_file)
            df = pd.DataFrame(data)

            # Convertir bytes a string si es necesario
            for col in df.select_dtypes([object]):
                df[col] = df[col].apply(lambda x: x.decode('utf-8'))

            # Reducir dataset para usar menos memoria
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)

            # Dividir dataset
            train, temp = train_test_split(df, test_size=0.4, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)

            # Graficar solo 4 columnas numéricas
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4]

            images = []
            for col in numeric_cols:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(df[col], bins=20, color='skyblue', edgecolor='black')
                ax.set_title(col)
                ax.set_xlabel('Valor')
                ax.set_ylabel('Frecuencia')

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight')
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plt.close(fig)

                images.append(base64.b64encode(image_png).decode('utf-8'))

            # Renderizar solo las gráficas
            return render(request, 'index.html', {'images': images})

        except Exception as e:
            return render(request, 'index.html', {'error': f"Ocurrió un error: {str(e)}"})

    return render(request, 'index.html')
