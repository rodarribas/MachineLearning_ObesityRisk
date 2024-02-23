# ML_Obesity_Risk
Proyecto de Machine Learning para predecir el riesgo de obesidad en base a un dataset de Kaggle
Se proponen 3 modelos:
- Modelo 1: el mejor modelo, basado en la selección de variables de acuerdo a un heatmap, logaritmo, escalado MinMax, grid search de los 3 mejores modelos resultantes del baseline y encapsulación en un voting ensemble.
- Modelo 2: se hizo un undersampling manual de la target más numerosa basado en el EDA. Posteriormente un RandomOversampling para equilibrar totalmente la target y finalmente un PCA. Tras aplicar LGBM al dataset resultante, no se obtuvieron mejores resultados que en el Modelo_1 (en fase baseline).
- Modelo 3: basado únicamente en la información proporcionada por el "negocio". Se tuvo en cuenta sólo el IMC y la frecuencia de actividad física. Se aplicaron los mismos modelos que en el Modelo 1 (con sus hiperparámetros adaptados) y se encapsularon en un voting ensemble. Resultado: el modelo es más ligero, pero no más preciso que el Modelo_1
