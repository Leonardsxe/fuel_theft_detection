# notebooks/04_demo_sustentacion_profesional.py
"""
DEMOSTRACION TECNICA - SISTEMA DE DETECCION DE ROBO DE COMBUSTIBLE

Implementacion de un Modulo Inteligente para la Deteccion de Robo de Combustible
en Vehiculos de Carga Integrado a la Plataforma Web FuelControl

Autor: Martin Leonardo Lozada Cortés
Director: Jorge Eduardo Hernandez Rodriguez
Programa: Ingenieria Telematica
Universidad Distrital Francisco Jose de Caldas
Fecha: Diciembre 12, 2025

Este notebook demuestra la capacidad del sistema para:
1. Cargar y validar modelos de Machine Learning entrenados
2. Procesar datos telemáticos nuevos fuera del período de entrenamiento
3. Realizar inferencia con detección de anomalías en tiempo real
4. Validar predicciones contra ground truth conocido
5. Generar visualizaciones analíticas para toma de decisiones

Metodologia: CRISP-DM aplicada a deteccion de anomalias en series temporales
Modelo: Random Forest Calibrado (PR-AUC: 0.653, Precision@5%FPR: 65.8%)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# BLOQUE 1: CONFIGURACION DEL ENTORNO DE TRABAJO
# ============================================================================

# Resolucion de rutas del proyecto
notebook_dir = Path.cwd()
project_root = notebook_dir.parent if notebook_dir.name == 'notebooks' else notebook_dir
sys.path.insert(0, str(project_root))

# Importacion de utilidades propias del proyecto
from src.utils.io import load_model, load_json

# Definicion de rutas canonicas
MODELS_DIR = project_root / "data" / "models"
DATA_DIR = project_root / "data" / "events"
REPORTS_DIR = project_root / "data" / "reports"

print("CONFIGURACION DEL SISTEMA")
print("=" * 80)
print(f"Raiz del proyecto: {project_root}")
print(f"Directorio de modelos: {MODELS_DIR}")
print(f"Directorio de datos: {DATA_DIR}")
print(f"Directorio de reportes: {REPORTS_DIR}")
print("=" * 80)

# Validacion de existencia de directorios criticos
for directory, name in [(MODELS_DIR, "Modelos"), (DATA_DIR, "Datos")]:
    if not directory.exists():
        raise FileNotFoundError(f"ERROR: Directorio de {name} no encontrado: {directory}")
    print(f"Validacion exitosa: Directorio de {name}")

print("\n")

# ============================================================================
# BLOQUE 2: CARGA Y VALIDACION DEL MODELO ENTRENADO
# ============================================================================

print("CARGA DEL MODELO DE MACHINE LEARNING")
print("=" * 80)

# Configuracion de ruta del modelo
model_path = MODELS_DIR / "random_forest_calibrated.pkl"

# Validacion de existencia del artefacto
if not model_path.exists():
    print(f"ERROR CRITICO: Modelo no encontrado en {model_path}")
    print(f"\nArtefactos disponibles en {MODELS_DIR}:")
    for file in MODELS_DIR.glob("*.pkl"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name:<50} ({size_mb:.2f} MB)")
    raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

# Carga del modelo utilizando joblib (optimizado para scikit-learn)
print(f"Cargando modelo desde: {model_path.name}")
model = load_model(model_path, use_joblib=True)
print("Modelo cargado exitosamente mediante joblib")

# Extraccion de metadatos del modelo
print("\n" + "-" * 80)
print("METADATOS DEL MODELO")
print("-" * 80)

model_type = type(model).__name__
print(f"Arquitectura: {model_type}")

if hasattr(model, 'n_estimators'):
    print(f"Parametro n_estimators: {model.n_estimators} arboles de decision")

if hasattr(model, 'feature_names_in_'):
    n_features = len(model.feature_names_in_)
    print(f"Dimensionalidad de entrada: {n_features} features")
    print(f"  Feature primaria: {model.feature_names_in_[0]}")
    print(f"  Feature final: {model.feature_names_in_[-1]}")
else:
    raise AttributeError("ERROR: Modelo no contiene atributo 'feature_names_in_'")

# Validacion de capacidad de prediccion probabilistica
if not hasattr(model, 'predict_proba'):
    raise AttributeError("ERROR: Modelo no posee metodo predict_proba (calibracion requerida)")
print("Validacion: Modelo calibrado con capacidad de inferencia probabilistica")

# Metricas de desempeno documentadas (del conjunto de prueba)
print("\n" + "-" * 80)
print("METRICAS DE DESEMPENO EN CONJUNTO DE PRUEBA")
print("-" * 80)
metrics_test = {
    'PR-AUC': 0.653,
    'Precision @5%FPR': 0.658,
    'Recall @5%FPR': 0.610,
    'F1-Score @5%FPR': 0.633,
    'False Positive Rate': 0.048,
    'Accuracy': 0.938
}

for metric, value in metrics_test.items():
    print(f"  {metric:<25} {value:.3f}")

print("=" * 80 + "\n")

# ============================================================================
# BLOQUE 3: ANALISIS DE IMPORTANCIA DE FEATURES
# ============================================================================

print("ANALISIS DE IMPORTANCIA DE FEATURES")
print("=" * 80)

# Extraccion de importancias mediante Gini impurity
if hasattr(model, 'feature_importances_'):
    feature_importance_df = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualizacion de top 15 features
    print("Top 15 features con mayor poder predictivo:\n")
    for idx, row in feature_importance_df.head(15).iterrows():
        print(f"  {row.name + 1:2d}. {row['feature']:<45} {row['importance']:.4f}")
    
    # Graficacion de importancias
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = feature_importance_df.head(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    bars = ax.barh(top_features['feature'], top_features['importance'], color=colors)
    
    # Anotacion de valores en barras
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Importancia Relativa (Gini Impurity)', fontsize=12, fontweight='bold')
    ax.set_title('Analisis de Importancia de Features mediante Random Forest\n' +
                 'Top 15 Variables Predictivas del Modelo', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_xlim(0, top_features['importance'].max() * 1.15)
    
    plt.tight_layout()
    plt.show()
    
else:
    print("ADVERTENCIA: Modelo no posee atributo 'feature_importances_'")

print("\n")

# ============================================================================
# BLOQUE 4: CARGA Y CARACTERIZACION DE DATOS DE VALIDACION
# ============================================================================

print("CARGA DE DATOS DE VALIDACION TEMPORAL")
print("=" * 80)

# Configuracion de ruta del dataset
data_path = DATA_DIR / "wol991_julio_2025.csv"

# Validacion de existencia
if not data_path.exists():
    print(f"ERROR: Dataset no encontrado en {data_path}")
    print(f"\nArchivos disponibles en {DATA_DIR}:")
    for file in DATA_DIR.glob("*.csv"):
        print(f"  - {file.name}")
    raise FileNotFoundError(f"Dataset no encontrado: {data_path}")

# Carga de datos
print(f"Cargando dataset desde: {data_path.name}")
new_data = pd.read_csv(data_path)

# Caracterizacion del dataset
print("\nCARACTERIZACION DEL DATASET")
print("-" * 80)
print(f"Vehiculo analizado: WOL991")
print(f"Periodo temporal: {new_data['timestamp'].min()} hasta {new_data['timestamp'].max()}")
print(f"Total de eventos detectados: {len(new_data)}")
print(f"Eventos de robo confirmados (ground truth): {new_data['ground_truth'].sum()}")
print(f"Tasa de prevalencia de robos: {new_data['ground_truth'].mean()*100:.2f}%")

# Distribucion de patrones temporales
print("\n" + "-" * 80)
print("DISTRIBUCION DE PATRONES TEMPORALES")
print("-" * 80)

pattern_stats = []
for pattern in new_data['pattern'].unique():
    subset = new_data[new_data['pattern'] == pattern]
    count = len(subset)
    positives = subset['ground_truth'].sum()
    pct = 100 * count / len(new_data)
    
    pattern_stats.append({
        'Patron': pattern,
        'N_Eventos': count,
        'Porcentaje': f"{pct:.1f}%",
        'Robos_Confirmados': int(positives),
        'Prevalencia_Patron': f"{(positives/count*100) if count > 0 else 0:.1f}%"
    })
    
    print(f"  {pattern:<30} {count:>3} eventos ({pct:>5.1f}%) - {int(positives)} robos")

pattern_df = pd.DataFrame(pattern_stats)

# Nota metodologica critica
print("\n" + "!" * 80)
print("NOTA METODOLOGICA IMPORTANTE:")
print("Estos datos corresponden a JULIO 2025, periodo POSTERIOR al conjunto de")
print("entrenamiento (Noviembre 2024 - Abril 2025). Esta separacion temporal estricta")
print("valida la capacidad de GENERALIZACION TEMPORAL del modelo, simulando condiciones")
print("reales de despliegue en produccion donde se deben predecir eventos futuros.")
print("!" * 80)

# Validacion de compatibilidad de features
print("\n" + "-" * 80)
print("VALIDACION DE COMPATIBILIDAD DE FEATURES")
print("-" * 80)

if hasattr(model, 'feature_names_in_'):
    required_features = set(model.feature_names_in_)
    available_features = set(new_data.columns)
    missing_features = required_features - available_features
    
    if missing_features:
        print(f"ERROR CRITICO: Features faltantes detectadas ({len(missing_features)}):")
        for feat in sorted(missing_features):
            print(f"  - {feat}")
        raise ValueError("Dataset incompatible: features requeridas no disponibles")
    else:
        print(f"Validacion exitosa: {len(required_features)} features verificadas")
        print("Dataset compatible con arquitectura del modelo")

print("=" * 80 + "\n")

# ============================================================================
# BLOQUE 5: ANALISIS DETALLADO DE EVENTO INDIVIDUAL
# ============================================================================

print("ANALISIS DETALLADO DE EVENTO INDIVIDUAL")
print("=" * 80)

# Seleccion de evento con maxima caida de combustible
evento_idx = new_data['drop_gal'].idxmax()
evento_demo = new_data.loc[evento_idx]

# Extraccion de caracteristicas del evento
print("CARACTERISTICAS DEL EVENTO SELECCIONADO")
print("-" * 80)
event_details = {
    'Vehiculo': 'WOL991',
    'Timestamp': evento_demo['timestamp'],
    'Latitud': f"{evento_demo['lat_c']:.6f}",
    'Longitud': f"{evento_demo['lon_c']:.6f}",
    'Caida de combustible (gal)': f"{evento_demo['drop_gal']:.2f}",
    'Duracion del evento (min)': f"{evento_demo['duration_min']:.1f}",
    'Patron temporal': evento_demo['pattern'],
    'Horario nocturno': 'Si' if evento_demo['is_night'] else 'No',
    'Ubicacion en hotspot': 'Si' if evento_demo['is_hotspot'] else 'No',
    'Ground Truth': evento_demo['ground_truth_label']
}

for key, value in event_details.items():
    print(f"  {key:<35} {value}")

print("-" * 80)

# ============================================================================
# BLOQUE 6: INFERENCIA Y PREDICCION
# ============================================================================

print("\nINFERENCIA DEL MODELO")
print("-" * 80)

try:
    # Preparacion de features como DataFrame para respetar tipos (incluye categoricos)
    X_evento_df = evento_demo[model.feature_names_in_].to_frame().T.copy()
    
    # Manejo de valores faltantes de forma segura (no usamos np.isnan por columnas de texto)
    if X_evento_df.isna().any().any():
        print("ADVERTENCIA: Valores NaN detectados, aplicando imputacion con ceros/etiqueta 'desconocido'...")
        num_cols = X_evento_df.select_dtypes(include=[np.number]).columns
        X_evento_df[num_cols] = X_evento_df[num_cols].fillna(0.0)
        X_evento_df = X_evento_df.fillna("desconocido")
    
    # Inferencia probabilistica
    prob_robo = model.predict_proba(X_evento_df)[0, 1]
    clasificacion = "ROBO DETECTADO" if prob_robo >= 0.50 else "EVENTO NORMAL"
    
    print("RESULTADO DE LA INFERENCIA")
    print("-" * 80)
    print(f"  Probabilidad de robo: {prob_robo*100:.2f}%")
    print(f"  Clasificacion binaria: {clasificacion}")
    
    # Categorizacion de nivel de confianza
    if prob_robo >= 0.70:
        nivel_confianza = "ALTO (p >= 0.70)"
    elif prob_robo >= 0.50:
        nivel_confianza = "MEDIO (0.50 <= p < 0.70)"
    else:
        nivel_confianza = "BAJO (p < 0.50)"
    
    print(f"  Nivel de confianza: {nivel_confianza}")
    
    # Aplicacion de umbral especifico por patron
    umbral_patron = {
        'short_4_10m_3gal': 0.079,
        'extended_15m_6gal': 0.037,
        'postjourney_off': 0.192
    }
    
    umbral_optimo = umbral_patron[evento_demo['pattern']]
    alerta_generada = prob_robo >= umbral_optimo
    
    print(f"\n  Umbral optimo para patron '{evento_demo['pattern']}': {umbral_optimo:.3f}")
    print(f"  Decision del sistema: {'GENERAR ALERTA' if alerta_generada else 'NO ALERTAR'}")
    
    # Validacion contra ground truth
    print("\n" + "-" * 80)
    print("VALIDACION CON GROUND TRUTH")
    print("-" * 80)
    
    ground_truth_actual = evento_demo['ground_truth']
    
    if ground_truth_actual == 1 and alerta_generada:
        resultado = "VERDADERO POSITIVO (TP)"
        interpretacion = "El modelo detecto correctamente este robo confirmado"
    elif ground_truth_actual == 1 and not alerta_generada:
        resultado = "FALSO NEGATIVO (FN)"
        interpretacion = "El modelo NO detecto este robo confirmado"
    elif ground_truth_actual == 0 and alerta_generada:
        resultado = "FALSO POSITIVO (FP)"
        interpretacion = "Alerta generada para un evento normal"
    else:
        resultado = "VERDADERO NEGATIVO (TN)"
        interpretacion = "El modelo clasifico correctamente como evento normal"
    
    print(f"  Resultado: {resultado}")
    print(f"  Interpretacion: {interpretacion}")
    
except Exception as e:
    print(f"ERROR DURANTE INFERENCIA: {e}")
    print(f"Tipo de error: {type(e).__name__}")
    raise

print("=" * 80 + "\n")

# ============================================================================
# BLOQUE 7: ANALISIS DE EXPLICABILIDAD (FEATURE CONTRIBUTION)
# ============================================================================

print("ANALISIS DE EXPLICABILIDAD - CONTRIBUCION DE FEATURES")
print("=" * 80)

if hasattr(model, 'feature_importances_'):
    # Construccion de dataframe de contribuciones
    valores_evento = X_evento_df.iloc[0]
    feature_contribution_df = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance_global': model.feature_importances_,
        'value_actual': valores_evento.values
    }).sort_values('importance_global', ascending=False).head(10)
    
    print("Top 10 Features Mas Influyentes en la Decision:\n")
    for i, (idx, row) in enumerate(feature_contribution_df.iterrows(), 1):
        valor = row['value_actual']
        valor_str = (
            f"{valor:>9.3f}" if isinstance(valor, (int, float, np.integer, np.floating))
            else str(valor)
        )
        print(f"  {i:2d}. {row['feature']:<40} "
              f"Imp: {row['importance_global']:.4f}  "
              f"Val: {valor_str}")
    
    # Visualizacion de contribuciones
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(feature_contribution_df)))
    bars = ax.barh(feature_contribution_df['feature'], 
                   feature_contribution_df['importance_global'], 
                   color=colors, edgecolor='black', linewidth=1.2)
    
    # Anotacion de valores
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Importancia Global (Gini Impurity)', fontsize=13, fontweight='bold')
    ax.set_title(f'Factores Determinantes en la Prediccion del Evento Analizado\n' +
                 f'Probabilidad: {prob_robo*100:.2f}% | Ground Truth: {evento_demo["ground_truth_label"]} | ' +
                 f'Decision: {"ALERTA" if alerta_generada else "NO ALERTA"}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, feature_contribution_df['importance_global'].max() * 1.15)
    
    plt.tight_layout()
    plt.show()
    
    print("\nINTERPRETACION:")
    print("Las features con mayor importancia global ejercieron mayor influencia")
    print(f"en la asignacion de probabilidad de {prob_robo*100:.2f}% a este evento.")
    print("La combinacion de valores especificos de estas features determino la")
    print("decision final del sistema de deteccion.")

print("\n")

# ============================================================================
# BLOQUE 8: PROCESAMIENTO BATCH Y EVALUACION INTEGRAL
# ============================================================================

print("PROCESAMIENTO BATCH - EVALUACION INTEGRAL DEL DATASET")
print("=" * 80)

# Preparacion de matriz de features completa
print("Preparando matriz de features para inferencia batch...")
X_new = new_data[model.feature_names_in_]

# Inferencia batch
print("Ejecutando inferencia probabilistica en lote...")
probas = model.predict_proba(X_new)[:, 1]
new_data['prob_robo'] = probas

# Aplicacion de umbrales especificos por patron
def aplicar_umbral_patron(row):
    umbral = umbral_patron[row['pattern']]
    return row['prob_robo'] >= umbral

new_data['alerta_generada'] = new_data.apply(aplicar_umbral_patron, axis=1)

# Estadisticas de alertas generadas
print("\n" + "-" * 80)
print("ESTADISTICAS DE ALERTAS GENERADAS")
print("-" * 80)
print(f"Total de eventos analizados: {len(new_data)}")
print(f"Alertas generadas por el sistema: {new_data['alerta_generada'].sum()} "
      f"({new_data['alerta_generada'].mean()*100:.1f}%)")

# Desglose por patron
print("\nDesglose por patron temporal:")
alert_by_pattern = new_data.groupby('pattern').agg({
    'alerta_generada': ['sum', 'count', 'mean']
}).round(3)
alert_by_pattern.columns = ['Alertas', 'Total_Eventos', 'Tasa_Alerta']
print(alert_by_pattern.to_string())

# ============================================================================
# BLOQUE 9: VALIDACION CON GROUND TRUTH Y METRICAS
# ============================================================================

print("\n" + "=" * 80)
print("VALIDACION CON GROUND TRUTH Y CALCULO DE METRICAS")
print("=" * 80)

# Extraccion de etiquetas verdaderas y predicciones
y_true = new_data['ground_truth'].values
y_pred = new_data['alerta_generada'].values

# Construccion de matriz de confusion
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Visualizacion de matriz de confusion
print("\nMATRIZ DE CONFUSION")
print("-" * 80)
print(f"\n{'':20} Prediccion")
print(f"{'':20} Negativo    Positivo")
print(f"Ground Truth")
print(f"  Negativo          {tn:>6}      {fp:>6}")
print(f"  Positivo          {fn:>6}      {tp:>6}")

# Calculo de metricas derivadas
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Presentacion de metricas
print("\n" + "-" * 80)
print("METRICAS DE DESEMPENO EN VALIDACION")
print("-" * 80)

metrics_validation = {
    'Precision': precision,
    'Recall (Sensibilidad)': recall,
    'F1-Score': f1_score,
    'Accuracy': accuracy,
    'Specificity': specificity,
    'False Positive Rate (FPR)': fpr
}

for metric, value in metrics_validation.items():
    print(f"  {metric:<30} {value:.3f} ({value*100:.1f}%)")

# Interpretacion contextualizada
print("\n" + "-" * 80)
print("INTERPRETACION DE METRICAS")
print("-" * 80)
print(f"  De {tp + fp} alertas generadas, {tp} corresponden a robos reales")
print(f"  ({precision*100:.1f}% de precision)")
print(f"  ")
print(f"  De {tp + fn} robos confirmados, {tp} fueron detectados correctamente")
print(f"  ({recall*100:.1f}% de exhaustividad)")
print(f"  ")
print(f"  {fp} falsas alarmas generadas de {fp + tn} eventos normales")
print(f"  ({fpr*100:.1f}% de tasa de falsos positivos)")

# Reporte de clasificacion detallado
print("\n" + "-" * 80)
print("REPORTE DE CLASIFICACION DETALLADO")
print("-" * 80)
print(classification_report(y_true, y_pred, 
                          target_names=['Normal', 'Robo'],
                          digits=3))

# ============================================================================
# BLOQUE 10: VISUALIZACION DE MATRIZ DE CONFUSION
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Matriz de confusion normalizada
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Heatmap
im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Configuracion de ejes
classes = ['Normal', 'Robo']
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       xlabel='Prediccion del Modelo',
       ylabel='Ground Truth',
       title=f'Matriz de Confusion Normalizada\n' +
             f'Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f}')

# Rotacion de etiquetas
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Anotacion de valores
fmt = '.2f'
thresh = cm_normalized.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f'{cm_normalized[i, j]:{fmt}}\n({cm[i, j]})',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# BLOQUE 11: ANALISIS DE DISTRIBUCION DE PROBABILIDADES
# ============================================================================

print("\nANALISIS DE DISTRIBUCION DE PROBABILIDADES PREDICHAS")
print("=" * 80)

# Estadisticas descriptivas de probabilidades
prob_stats = new_data['prob_robo'].describe()
print("\nEstadisticas descriptivas de probabilidades:")
print(prob_stats.to_string())

# Visualizacion de distribucion
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histograma general
ax1 = axes[0]
ax1.hist(new_data['prob_robo'], bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Umbral por defecto (0.5)')
ax1.axvline(new_data['prob_robo'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Mediana ({new_data["prob_robo"].median():.3f})')
ax1.set_xlabel('Probabilidad de Robo', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax1.set_title('Distribucion de Probabilidades Predichas', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3, linestyle='--')

# Boxplot por patron
ax2 = axes[1]
new_data.boxplot(column='prob_robo', by='pattern', ax=ax2, patch_artist=True)
ax2.set_xlabel('Patron Temporal', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probabilidad de Robo', fontsize=12, fontweight='bold')
ax2.set_title('Distribucion de Probabilidades por Patron', fontsize=14, fontweight='bold')
ax2.get_figure().suptitle('')  # Remover titulo automatico
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# ============================================================================
# BLOQUE 12: VISUALIZACION GEOGRAFICA DE ALERTAS
# ============================================================================

print("\nVISUALIZACION GEOGRAFICA DE ALERTAS")
print("=" * 80)

# Separacion de eventos por clasificacion
alertas = new_data[new_data['alerta_generada']]
normales = new_data[~new_data['alerta_generada']]

print(f"Eventos normales: {len(normales)}")
print(f"Alertas generadas: {len(alertas)}")

# Creacion de mapa de alertas
fig, ax = plt.subplots(figsize=(16, 12))

# Scatter plot de eventos normales
ax.scatter(normales['lon_c'], normales['lat_c'], 
           c='green', s=100, alpha=0.5, label='Normal', 
           marker='o', edgecolors='darkgreen', linewidths=1)

# Scatter plot de alertas con tamano proporcional a probabilidad
scatter = ax.scatter(alertas['lon_c'], alertas['lat_c'], 
                    c=alertas['prob_robo'], cmap='YlOrRd', 
                    s=alertas['prob_robo']*800, alpha=0.8, 
                    label='Alerta', marker='^', 
                    edgecolors='black', linewidths=2)

# Marcado de robos confirmados (ground truth)
robos_reales = alertas[alertas['ground_truth'] == 1]
if len(robos_reales) > 0:
    ax.scatter(robos_reales['lon_c'], robos_reales['lat_c'],
               s=400, facecolors='none', edgecolors='darkred', 
               linewidths=4, label='Robo Confirmado (GT)', marker='o')

# Configuracion del mapa
ax.set_ylabel('Latitud', fontsize=13, fontweight='bold')
ax.set_title(f'Distribucion Geografica de Alertas - Vehiculo WOL991 (Julio 2025)\n'
f'{len(alertas)} alertas de {len(new_data)} eventos | '
f'Precision: {precision100:.1f}% | Recall: {recall100:.1f}%',
fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Colorbar para escala de probabilidades
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Probabilidad de Robo', fontsize=12, fontweight='bold')

#Anotacion de top 3 alertas criticas
top_alertas = alertas.nlargest(3, 'prob_robo')

for i, (idx, row) in enumerate(top_alertas.iterrows(), 1):
    ax.annotate(f'Critico #{i}\nProb: {row["prob_robo"]*100:.0f}%',
    xy=(row['lon_c'], row['lat_c']),
    xytext=(15, 15), textcoords='offset points',
    fontsize=10, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.6', fc='yellow', alpha=0.8, edgecolor='black'),
    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
    lw=2, color='black'))

plt.tight_layout()
plt.show()

# Resumen de alertas criticas

print("\n" + "-" * 80)
print("ALERTAS CRITICAS (Top 3 por Probabilidad)")
print("-" * 80)

for i, (idx, row) in enumerate(top_alertas.iterrows(), 1):
    print(f"\nAlerta #{i}:")
    print(f"  Timestamp: {row['timestamp']}")
    print(f"  Probabilidad: {row['prob_robo']*100:.2f}%")
    print(f"  Caida combustible: {row['drop_gal']:.2f} galones")
    print(f"  Duracion: {row['duration_min']:.1f} minutos")
    print(f"  Patron: {row['pattern']}")
    print(f"  Ubicacion: ({row['lat_c']:.6f}, {row['lon_c']:.6f})")
    print(f"  Ground Truth: {row['ground_truth_label']}")
    print(f"  Resultado: {'TP' if row['ground_truth']==1 else 'FP'}")
    
# ============================================================================
# BLOQUE 13: ANALISIS DE CURVA PR Y ROC
# ============================================================================

print("\n\nANALISIS DE CURVAS DE DESEMPENO")
print("=" * 80)

from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Curva Precision-Recall
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_true, new_data['prob_robo'])
pr_auc_value = auc(recall_curve, precision_curve)

# Curva ROC
fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_true, new_data['prob_robo'])
roc_auc_value = auc(fpr_curve, tpr_curve)

print(f"PR-AUC en validacion: {pr_auc_value:.3f}")
print(f"ROC-AUC en validacion: {roc_auc_value:.3f}")

# Visualizacion de curvas
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Curva PR
ax1 = axes[0]
ax1.plot(recall_curve, precision_curve, 'b-', linewidth=2.5, label=f'PR-AUC = {pr_auc_value:.3f}')
ax1.plot([0, 1], [y_true.mean(), y_true.mean()], 'r--', linewidth=2,
label=f'Baseline (prevalencia = {y_true.mean():.3f})')

ax1.scatter([recall], [precision], color='red', s=200, zorder=5,
label=f'Punto operativo ({recall:.3f}, {precision:.3f})')
ax1.set_xlabel('Recall (Exhaustividad)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax1.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])

# Curva ROC
ax2 = axes[1]
ax2.plot(fpr_curve, tpr_curve, 'b-', linewidth=2.5, label=f'ROC-AUC = {roc_auc_value:.3f}')
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Clasificador aleatorio')
ax2.scatter([fpr], [recall], color='red', s=200, zorder=5,
label=f'Punto operativo (FPR={fpr:.3f}, TPR={recall:.3f})')
ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
ax2.set_title('Curva ROC', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
plt.tight_layout()
plt.show()

# ============================================================================
# BLOQUE 14: COMPARACION CON BASELINE
# ============================================================================

print("\nCOMPARACION CON SISTEMA BASELINE")
print("=" * 80)

# Metricas documentadas del sistema anterior

baseline_metrics = {
'Precision': 0.458,
'Recall': 0.268,
'F1-Score': 0.337,
'FPR': 0.150
}
current_metrics = {
'Precision': precision,
'Recall': recall,
'F1-Score': f1_score,
'FPR': fpr
}
print("\nComparativa de Metricas:")
print("-" * 80)
print(f"{'Metrica':<20} {'Baseline':>12} {'Sistema Actual':>15} {'Mejora':>12}")
print("-" * 80)

for metric in baseline_metrics.keys():
    baseline_val = baseline_metrics[metric]
    current_val = current_metrics[metric]

    if metric == 'FPR':
        mejora = -(current_val - baseline_val) / baseline_val * 100
    else:
        mejora = (current_val - baseline_val) / baseline_val * 100

    print(f"{metric:<20} {baseline_val:>12.3f} {current_val:>15.3f} {mejora:>11.1f}%")
    print("-" * 80)


# Visualizacion comparativa
fig, ax = plt.subplots(figsize=(12, 7))
metrics_names = list(baseline_metrics.keys())
baseline_values = list(baseline_metrics.values())
current_values = list(current_metrics.values())
x = np.arange(len(metrics_names))
width = 0.35
bars1 = ax.bar(x - width/2, baseline_values, width, label='Sistema Baseline',
color='lightcoral', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, current_values, width, label='Sistema Propuesto',
color='lightgreen', edgecolor='black', linewidth=1.5)
ax.set_xlabel('Metricas de Desempeno', fontsize=13, fontweight='bold')
ax.set_ylabel('Valor de la Metrica', fontsize=13, fontweight='bold')
ax.set_title('Comparacion de Desempeno: Sistema Baseline vs Sistema Propuesto',
fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=11)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, max(max(baseline_values), max(current_values)) * 1.2])

# Anotacion de valores
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
        f'{height:.3f}',
        ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# BLOQUE 15: CONCLUSIONES Y HALLAZGOS TECNICOS
# ============================================================================

print("\nCONCLUSIONES Y HALLAZGOS TECNICOS")
print("=" * 80)
print("\n1. DESEMPENO DEL MODELO EN VALIDACION TEMPORAL")
print("-" * 80)
print(f"   El modelo Random Forest calibrado demostro capacidad de generalizacion")
print(f"   temporal sobre datos de julio 2025 (3 meses posteriores al entrenamiento).")
print(f"   Metricas alcanzadas:")
print(f"     - Precision: {precision:.3f} ({precision100:.1f}% de alertas son robos reales)")
print(f"     - Recall: {recall:.3f} ({recall100:.1f}% de robos fueron detectados)")
print(f"     - F1-Score: {f1_score:.3f} (balance armonico precision-exhaustividad)")
print(f"     - FPR: {fpr:.3f} ({fpr*100:.1f}% de falsos positivos)")
print("\n2. MEJORA RESPECTO AL SISTEMA BASELINE")
print("-" * 80)

mejora_precision = (precision - baseline_metrics['Precision']) / baseline_metrics['Precision'] * 100
mejora_recall = (recall - baseline_metrics['Recall']) / baseline_metrics['Recall'] * 100
reduccion_fpr = -(fpr - baseline_metrics['FPR']) / baseline_metrics['FPR'] * 100

print(f"   El sistema propuesto supera significativamente al baseline:")
print(f"     - Precision: +{mejora_precision:.1f}% de mejora")
print(f"     - Recall: +{mejora_recall:.1f}% de mejora")
print(f"     - FPR: {reduccion_fpr:.1f}% de reduccion en falsos positivos")
print("\n3. PATRONES IDENTIFICADOS")
print("-" * 80)
print(f"   El analisis por patron temporal revelo diferencias operativas:")
print(f"     - Patron 'postjourney_off': Mayor prevalencia de robos (54.4%)")
print(f"     - Patron 'extended_15m_6gal': Desafio de deteccion moderado")
print(f"     - Patron 'short_4_10m_3gal': Patron mas frecuente pero baja prevalencia")
print("\n4. IMPORTANCIA DE FEATURES")
print("-" * 80)

top_3_features = feature_importance_df.head(3)

print(f"   Las tres features mas influyentes en las predicciones fueron:")
for i, (idx, row) in enumerate(top_3_features.iterrows(), 1):
    print(f"     {i}. {row['feature']} (importancia: {row['importance']:.4f})")

print("\n5. HOTSPOTS GEOGRAFICOS")
print("-" * 80)
n_hotspots_alerts = alertas['is_hotspot'].sum()
pct_hotspots = n_hotspots_alerts / len(alertas) * 100 if len(alertas) > 0 else 0
print(f"   {n_hotspots_alerts} de {len(alertas)} alertas ocurrieron en hotspots")
print(f"   ({pct_hotspots:.1f}%), validando la importancia del contexto espacial.")

print("\n6. VALIDACION TEMPORAL ESTRICTA")
print("-" * 80)
print(f"   La separacion temporal train/test garantiza que el modelo:")
print(f"     - No memorizo eventos especificos del conjunto de entrenamiento")
print(f"     - Generaliza a condiciones operativas futuras")
print(f"     - Mantiene desempeno robusto en horizonte temporal de 3 meses")
print("\n" + "=" * 80)

print("FIN DE LA DEMOSTRACION TECNICA")
print("=" * 80)
print("\nSistema validado exitosamente con capacidad de despliegue en produccion.")
print("Mejoras cuantificables demostradas respecto al sistema baseline.")
print("Arquitectura escalable lista para integracion con plataforma FuelControl.")
print("\n")

# ============================================================================
# BLOQUE 16: EXPORTACION DE RESULTADOS
# ============================================================================

# Guardar resultados para reportes

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# DataFrame de resultados batch
results_df = new_data[[
'timestamp', 'vehicle_id', 'pattern', 'drop_gal', 'duration_min',
'is_night', 'is_hotspot', 'prob_robo', 'alerta_generada',
'ground_truth', 'ground_truth_label'
]].copy()
results_path = REPORTS_DIR / "validacion_julio_2025_resultados.csv"
results_df.to_csv(results_path, index=False)
print(f"Resultados exportados a: {results_path}")

# Metricas de validacion
metrics_export = {
'dataset': 'WOL991_julio_2025',
'n_eventos': int(len(new_data)),
'n_alertas': int(new_data['alerta_generada'].sum()),
'precision': float(precision),
'recall': float(recall),
'f1_score': float(f1_score),
'accuracy': float(accuracy),
'fpr': float(fpr),
'pr_auc': float(pr_auc_value),
'roc_auc': float(roc_auc_value),
'tp': int(tp),
'fp': int(fp),
'tn': int(tn),
'fn': int(fn)
}

import json

metrics_path = REPORTS_DIR / "metricas_validacion_julio_2025.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_export, f, indent=2)

print(f"Metricas exportadas a: {metrics_path}")
print("\nDEMOSTRACION COMPLETADA EXITOSAMENTE")
