import numpy as np
import csv
import time

def carregar_e_codificar_dados(caminho_arquivo):
    X_raw = []
    y_raw = []
    
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        leitor = csv.reader(f)
        try:
            cabecalho = next(leitor)
        except:
            pass
        
        for linha in leitor:
            if len(linha) > 1:
                X_raw.append(linha[:-1])
                y_raw.append(linha[-1])
                
    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    
    # Convertendo textos para números (Label Encoding na raça)
    X_codificado = np.zeros(X_raw.shape)
    for col in range(X_raw.shape[1]):
        try:
            # Tenta converter direto para float (se já for número)
            X_codificado[:, col] = X_raw[:, col].astype(float)
        except ValueError:
            # Se for texto, mapeia cada palavra única para um número inteiro
            valores_unicos = np.unique(X_raw[:, col])
            mapa = {valor: i for i, valor in enumerate(valores_unicos)}
            X_codificado[:, col] = [mapa[v] for v in X_raw[:, col]]
            
    # Codificando a classe alvo (y)
    classes_unicas = np.unique(y_raw)
    mapa_y = {valor: i for i, valor in enumerate(classes_unicas)}
    y_codificado = np.array([mapa_y[v] for v in y_raw])
    
    return X_codificado, y_codificado

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

def calcular_metricas(y_real, y_pred):
    # Classe positiva será 1, classe negativa será 0
    VP = np.sum((y_real == 1) & (y_pred == 1))
    VN = np.sum((y_real == 0) & (y_pred == 0))
    FP = np.sum((y_real == 0) & (y_pred == 1))
    FN = np.sum((y_real == 1) & (y_pred == 0))
    
    acuracia = (VP + VN) / len(y_real)
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0.0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0.0
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0.0
    
    return acuracia, precisao, f1_score

# =====================================================================
# ALGORITMOS DE APRENDIZADO DE MÁQUINA
# =====================================================================

class KNN_Classificador:
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
            
            # Pega os indices dos k vizinhos mais próximos
            k_indices = np.argsort(distancias)[:self.k]
            k_classes = self.y_train[k_indices]
            
            # Votação majoritária
            valores_unicos, contagens = np.unique(k_classes, return_counts=True)
            classe_vencedora = valores_unicos[np.argmax(contagens)]
            y_pred.append(classe_vencedora)
            
        return np.array(y_pred)

class NaiveBayesUnivariado:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.parametros = {}
        
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.parametros[c] = {
                'media': np.mean(X_c, axis=0),
                'std': np.std(X_c, axis=0) + 1e-6, # +1e-6 para evitar divisão por zero
                'prior': len(X_c) / len(X_train)
            }

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            posteriors = []
            for c in self.classes:
                media = self.parametros[c]['media']
                std = self.parametros[c]['std']
                prior = self.parametros[c]['prior']
                
                # Fórmula da Gaussiana Univariada
                pdf = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - media)**2) / (2 * std**2))
                
                # log para evitar underflow matemático em números muito pequenos
                probabilidade = np.sum(np.log(pdf)) + np.log(prior)
                posteriors.append(probabilidade)
                
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

class NaiveBayesMultivariado:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.parametros = {}
        
        for c in self.classes:
            X_c = X_train[y_train == c]
            media = np.mean(X_c, axis=0)
            # Matriz de covariancia usando np.cov como pedido.
            # Somamos um 'epsilon' à diagonal para garantir que a matriz não seja singular (invertível)
            cov = np.cov(X_c, rowvar=False) + np.eye(X_train.shape[1]) * 1e-4 
            
            self.parametros[c] = {
                'media': media,
                'cov_inv': np.linalg.inv(cov),
                'cov_det': np.linalg.det(cov),
                'prior': len(X_c) / len(X_train)
            }

    def predict(self, X_test):
        y_pred = []
        d = X_test.shape[1]
        
        for x in X_test:
            posteriors = []
            for c in self.classes:
                media = self.parametros[c]['media']
                cov_inv = self.parametros[c]['cov_inv']
                cov_det = self.parametros[c]['cov_det']
                prior = self.parametros[c]['prior']
                
                diff = x - media
                if cov_det <= 0: cov_det = 1e-10 # Proteção
                
                # Fórmula da Normal Multivariada
                expoente = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
                coeficiente = 1 / (((2 * np.pi)**(d/2)) * np.sqrt(cov_det))
                
                probabilidade = np.log(coeficiente) + expoente + np.log(prior)
                posteriors.append(probabilidade)
                
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

# =====================================================================
# 3. EXECUÇÃO DA VALIDAÇÃO CRUZADA E IMPRESSÃO DOS RESULTADOS
# =====================================================================

def executar_experimentos():
    print("Carregando e processando os dados...")
    try:
        X, y = carregar_e_codificar_dados('german_credit.csv')
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    folds = k_fold_split(X, k=5)
    
    modelos = {
        "K-vizinhos (Dist. Euclidiana)": KNN_Classificador(k=5, distancia='euclidiana'),
        "K-vizinhos (Dist. Manhattan)": KNN_Classificador(k=5, distancia='manhattan'),
        "Bayesiano (Univariado)": NaiveBayesUnivariado(),
        "Bayesiano (Multivariado)": NaiveBayesMultivariado()
    }
    
    resultados_finais = {}
    
    for nome, modelo in modelos.items():
        acc_lista, prec_lista, f1_lista = [], [], []
        tempo_treino_lista, tempo_teste_lista = [], []
        
        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Tempo de Treino
            inicio_treino = time.time()
            modelo.fit(X_train, y_train)
            fim_treino = time.time()
            
            # Tempo de Teste
            inicio_teste = time.time()
            y_pred = modelo.predict(X_test)
            fim_teste = time.time()
            
            # Métricas
            acc, prec, f1 = calcular_metricas(y_test, y_pred)
            
            acc_lista.append(acc)
            prec_lista.append(prec)
            f1_lista.append(f1)
            tempo_treino_lista.append(fim_treino - inicio_treino)
            tempo_teste_lista.append(fim_teste - inicio_teste)
            
        # Salvando médias e desvios
        resultados_finais[nome] = {
            "Acurácia": f"{np.mean(acc_lista):.2f} ± {np.std(acc_lista):.2f}",
            "Precisão": f"{np.mean(prec_lista):.2f} ± {np.std(prec_lista):.2f}",
            "F1-Score": f"{np.mean(f1_lista):.2f} ± {np.std(f1_lista):.2f}",
            "T. Treino(s)": f"{np.mean(tempo_treino_lista):.4f} ± {np.std(tempo_treino_lista):.4f}",
            "T. Teste(s)": f"{np.mean(tempo_teste_lista):.4f} ± {np.std(tempo_teste_lista):.4f}",
        }

    # Imprimindo a tabela 
    print("\n" + "="*105)
    print(f"{'Classificador':<30} | {'Acurácia':<13} | {'Precisão':<13} | {'F1-Score':<13} | {'T. Treino (s)':<14} | {'T. Teste (s)':<14}")
    print("-" * 105)
    
    for nome, stats in resultados_finais.items():
        print(f"{nome:<30} | {stats['Acurácia']:<13} | {stats['Precisão']:<13} | {stats['F1-Score']:<13} | {stats['T. Treino(s)']:<14} | {stats['T. Teste(s)']:<14}")
    print("="*105)

if __name__ == "__main__":
    executar_experimentos()