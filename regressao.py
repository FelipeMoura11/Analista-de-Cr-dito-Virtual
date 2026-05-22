import numpy as np
import csv
import time

def carregar_e_codificar_dados(caminho_arquivo):
    X_raw = []
    y_raw = []
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        leitor = csv.reader(f)
        try:
            next(leitor) 
        except:
            pass
        
        for linha in leitor:
            if len(linha) > 1:
                X_raw.append(linha[:-1])
                y_raw.append(linha[-1])
                
    X_raw = np.array(X_raw)
    y = np.array(y_raw, dtype=float) # O Alvo (Purchase) é número contínuo
    
    # Label Encoding para converter as variáveis em texto para números
    X_codificado = np.zeros(X_raw.shape)
    for col in range(X_raw.shape[1]):
        try:
            X_codificado[:, col] = X_raw[:, col].astype(float)
        except ValueError:
            valores_unicos = np.unique(X_raw[:, col])
            mapa = {valor: i for i, valor in enumerate(valores_unicos)}
            X_codificado[:, col] = [mapa[v] for v in X_raw[:, col]]
            
    return X_codificado, y

def k_fold_split(X, k=5, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    tamanho_fold = len(X) // k
    folds = []
    
    for i in range(k):
        inicio = i * tamanho_fold
        fim = (i + 1) * tamanho_fold if i != k - 1 else len(X)
        test_idx = indices[inicio:fim]
        train_idx = np.concatenate((indices[:inicio], indices[fim:]))
        folds.append((train_idx, test_idx))
        
    return folds

# =====================================================================
# 2. ALGORITMOS DE REGRESSÃO
# =====================================================================

class RegressaoLinearMultipla:
    def fit(self, X_train, y_train):
        # Coluna de 1s para o intercepto
        X_b = np.column_stack((np.ones(len(X_train)), X_train))
        # Usamos pseudo-inversa (pinv) que é mais segura contra matrizes singulares
        self.beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_train

    def predict(self, X_test):
        X_b = np.column_stack((np.ones(len(X_test)), X_test))
        return X_b @ self.beta

class KNN_Regressor:
    def __init__(self, k=5, distancia='euclidiana'):
        self.k = k
        self.distancia = distancia

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            if self.distancia == 'euclidiana':
                distancias = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            elif self.distancia == 'manhattan':
                distancias = np.sum(np.abs(self.X_train - x), axis=1)
            
            # Ordena e pega os índices dos k mais próximos
            k_indices = np.argsort(distancias)[:self.k]
            # Pega os valores da variável Y correspondentes
            k_valores = self.y_train[k_indices]
            
            # Na regressão, o KNN tira a média dos vizinhos
            y_pred.append(np.mean(k_valores))
            
        return np.array(y_pred)

# =====================================================================
# 3. EXECUÇÃO E AVALIAÇÃO
# =====================================================================

def executar_regressao():
    print("Carregando e processando os dados de regressão...")
    try:
        X, y = carregar_e_codificar_dados('black_friday.csv') 
        
        X = X[:3000] 
        y = y[:3000]
        
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    folds = k_fold_split(X, k=5)
    
    modelos = {
        "Regressão Linear Múltipla": RegressaoLinearMultipla(),
        "K-vizinhos (Dist. Euclidiana)": KNN_Regressor(k=5, distancia='euclidiana'),
        "K-vizinhos (Dist. Manhattan)": KNN_Regressor(k=5, distancia='manhattan')
    }
    
    resultados_finais = {}
    p = X.shape[1] # Número de variáveis independentes para o R2 Ajustado
    
    for nome, modelo in modelos.items():
        r2_lista, r2_adj_lista = [], []
        tempo_treino_lista, tempo_teste_lista = [], []
        
        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            inicio_treino = time.time()
            modelo.fit(X_train, y_train)
            fim_treino = time.time()
            
            inicio_teste = time.time()
            y_pred = modelo.predict(X_test)
            fim_teste = time.time()
            
            # Cálculos de R2 e R2 Ajustado
            SQE = np.sum((y_test - y_pred)**2)
            SQT = np.sum((y_test - np.mean(y_test))**2)
            r2 = 1 - (SQE/SQT) if SQT != 0 else 0
            
            n_test = len(y_test)
            r2_adj = 1 - ((1 - r2) * (n_test - 1) / (n_test - p - 1)) if (n_test - p - 1) > 0 else 0
            
            r2_lista.append(r2)
            r2_adj_lista.append(r2_adj)
            tempo_treino_lista.append(fim_treino - inicio_treino)
            tempo_teste_lista.append(fim_teste - inicio_teste)
            
        resultados_finais[nome] = {
            "R2-Score": f"{np.mean(r2_lista):.4f} ± {np.std(r2_lista):.4f}",
            "R2 Ajustado": f"{np.mean(r2_adj_lista):.4f} ± {np.std(r2_adj_lista):.4f}",
            "T. Treino (s)": f"{np.mean(tempo_treino_lista):.4f} ± {np.std(tempo_treino_lista):.4f}",
            "T. Teste (s)": f"{np.mean(tempo_teste_lista):.4f} ± {np.std(tempo_teste_lista):.4f}"
        }

    # Imprimindo a tabela formatada
    print("\n" + "="*100)
    print(f"{'Regressor':<30} | {'R2-Score':<18} | {'R2 Ajustado':<18} | {'T. Treino (s)':<12} | {'T. Teste (s)':<12}")
    print("-" * 100)
    for nome, stats in resultados_finais.items():
        print(f"{nome:<30} | {stats['R2-Score']:<18} | {stats['R2 Ajustado']:<18} | {stats['T. Treino (s)']:<12} | {stats['T. Teste (s)']:<12}")
    print("="*100)

if __name__ == "__main__":
    executar_regressao()