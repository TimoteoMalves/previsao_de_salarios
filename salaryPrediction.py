import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# --- CONFIGURAÇÕES E VARIÁVEIS GLOBAIS ---
dataset_path = "C:\\Users\\timot\\OneDrive\\Área de Trabalho\\Aulas 2025\\inteligencia artificial 2\\source\\static\\dataset.csv"
TARGET_ORIGINAL = 'salary_in_usd'
TARGET_CLASS = 'salary_class' # Novo target binário
TOP_N_JOBS = 15
MIN_SALARY = 5000
MAX_SALARY = 500000

# --- 1. Load Data, Outlier Management e DISCRETIZAÇÃO ---
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Erro: O arquivo {dataset_path} não foi encontrado.")
    exit()

# A. Outlier Management (Remoção)
df_model = df[
    (df[TARGET_ORIGINAL] >= MIN_SALARY) & 
    (df[TARGET_ORIGINAL] <= MAX_SALARY)
].copy()

# B. DISCRETIZAÇÃO: Criando a Classe Salarial
# Usamos o 80/20 temporário APENAS para calcular a mediana corretamente
df_train_temp, _ = train_test_split(df_model, test_size=0.2, random_state=42)
SALARY_THRESHOLD = df_train_temp[TARGET_ORIGINAL].median()

# Cria a nova coluna Target binária: 1 (Alto) se > Limite, 0 (Baixo) se <= Limite
df_model[TARGET_CLASS] = (df_model[TARGET_ORIGINAL] > SALARY_THRESHOLD).astype(int)

print(f"\n--- Estratégia de Classificação ---")
print(f"Salários acima de ${SALARY_THRESHOLD:,.2f} são classificados como 'Alto'.")
print(f"Target: Prever se o salário é ALTO (1) ou BAIXO (0).")

# C. Feature Engineering (Mantido igual, crucial para o modelo)
job_counts = df_model['job_title'].value_counts()
top_jobs_list = job_counts.index[:TOP_N_JOBS].tolist()
df_model['job_title_grouped'] = df_model['job_title'].apply(
    lambda x: x if x in top_jobs_list else 'Other_Job_Title'
)
US_COUNTRIES = ['US', 'USA']
df_model['is_usa_company'] = df_model['company_location'].apply(
    lambda x: 1 if x in US_COUNTRIES else 0
)
df_model['exp_level_usa'] = df_model['experience_level'] + "_" + df_model['is_usa_company'].astype(str)
df_model['remote_ratio'] = df_model['remote_ratio'].astype(str)

# D. Seleção Final e One-Hot Encoding
FINAL_FEATURES = [
    'work_year', 'is_usa_company', 'remote_ratio', 'company_size', 
    'job_title_grouped', 'exp_level_usa'
]
df_encoded = pd.get_dummies(df_model[FINAL_FEATURES + [TARGET_CLASS]], 
                            columns=['remote_ratio', 'company_size', 'job_title_grouped', 'exp_level_usa'], 
                            drop_first=True)

X = df_encoded.drop(columns=[TARGET_CLASS])
y = df_encoded[TARGET_CLASS]

# --- 2. Splitting Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {len(X_train)} | Testing set size: {len(X_test)}")
print(f"Total de features: {X.shape[1]}")

# --- 3. Hyperparameter Tuning (Grid Search para Classificação) ---
# Adiciona mais combinações de hiperparâmetros
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

param_grid = {
    'n_estimators': [100, 200, 300],         
    'max_depth': [15, 25, 35],            
    'min_samples_split': [2, 5, 10],      
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# Usamos StratifiedKFold para garantir que cada fold de validação
# tenha a mesma proporção das classes '0' e '1'. Isso é crucial para dados desbalanceados.
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=rf_base, 
    param_grid=param_grid, 
    cv=stratified_kfold,                 
    scoring='f1', # Otimiza pelo F1-score (equilíbrio entre precisão e recall)
    verbose=0, 
    n_jobs=-1             
)

print("\n--- INICIANDO GRID SEARCH PARA CLASSIFICAÇÃO ---")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("Tuning completo. Melhores parâmetros encontrados:")
print(grid_search.best_params_)


# --- 4. Prediction and Evaluation ---
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


print("\n--- AVALIAÇÃO DO MODELO DE CLASSIFICAÇÃO ---")
print(f"Acurácia Geral: {accuracy:.4f}")
print("\n--- Matriz de Confusão ---")
print(conf_matrix)
print("\n--- Relatório de Classificação ---")
print(report)

# --- 5. Visualizações ---
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Baixo (0)', 'Alto (1)'], yticklabels=['Baixo (0)', 'Alto (1)'])
plt.title('Matriz de Confusão')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.show()

importances = best_model.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 7))
sns.barplot(x=forest_importances, y=forest_importances.index)
plt.title('Top 15 Feature Importance (Classificação)')
plt.xlabel('Importância Relativa')
plt.tight_layout()
plt.show()