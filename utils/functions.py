import seaborn as sns

# Función para determinar el mínimo y máximo de una distribución de acuerdo al boxplot
def iqr_fence(x):
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1
    Lower_Fence = Q1 - (1.5 * IQR)
    Upper_Fence = Q3 + (1.5 * IQR)
    l = min(x[x>Lower_Fence])
    u = max(x[x<Upper_Fence])
    return [l,u]

# Ploteo de gráfico de barras con números
def conteo(df, variable, hue=None, order=None, stat='count'):
    ax = sns.countplot(data=df, x=variable, hue=hue, stat=stat, order=order)
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    return ax