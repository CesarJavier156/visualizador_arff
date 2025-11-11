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
            arff_file = request.FILES['file']
            data, meta = arff.loadarff(arff_file)
            df = pd.DataFrame(data)

            # Convertir columnas tipo bytes
            for col in df.select_dtypes([object]):
                try:
                    df[col] = df[col].apply(lambda x: x.decode('utf-8'))
                except Exception:
                    pass

            # Limitar tamaño del dataset
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)

            # Separar
            train, temp = train_test_split(df, test_size=0.4, random_state=42)
            val, test = train_test_split(temp, test_size=0.5, random_state=42)

            # Seleccionar columnas numéricas
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4]
            if len(numeric_cols) == 0:
                return render(request, 'index.html', {
                    'error': 'El archivo no contiene columnas numéricas para graficar.'
                })

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

            return render(request, 'index.html', {'images': images})

        except Exception as e:
            print("⚠️ ERROR DETECTADO EN EL SERVIDOR:", e)
            return render(request, 'index.html', {'error': f"Ocurrió un error en el servidor: {str(e)}"})

    return render(request, 'index.html')
