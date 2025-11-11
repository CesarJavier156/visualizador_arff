import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Evita errores en Render
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from liac_arff import load
from io import StringIO

def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('file'):
        try:
            file = request.FILES['file']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            uploaded_file_path = fs.path(filename)

            # Leer el archivo ARFF
            with open(uploaded_file_path, 'r', encoding='utf-8') as f:
                dataset = load(f)
            df = pd.DataFrame(dataset['data'], columns=[a[0] for a in dataset['attributes']])

            # Guardar el contenido para mostrarlo en la página
            with open(uploaded_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Dividir datos en conjuntos
            from sklearn.model_selection import train_test_split
            train_set, temp_set = train_test_split(df, test_size=0.4, random_state=42)
            validation_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

            # Graficar
            plt.figure(figsize=(8, 5))
            df.hist(figsize=(8, 5))
            graph_path = os.path.join('media', 'graph.png')
            plt.savefig(graph_path)
            plt.close()

            context.update({
                'columns': df.columns,
                'df_html': df.to_html(classes="table table-striped", index=False),
                'train_shape': train_set.shape,
                'validation_shape': validation_set.shape,
                'test_shape': test_set.shape,
                'file_content': file_content,
                'graph_path': '/' + graph_path,
            })

        except Exception as e:
            context['error'] = f"Ocurrió un error al procesar el archivo: {e}"

    return render(request, 'index.html', context)
