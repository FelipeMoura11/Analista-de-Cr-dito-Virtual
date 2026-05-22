# 💳 Previsão de Risco de Crédito (German Credit Data)
> **Construindo Machine Learning do Zero (From Scratch) com Python e NumPy**

## 🎯 O Problema de Negócio
Bancos recebem milhares de pedidos de crédito diariamente. Analisar fichas manualmente é um processo lento e sujeito a viés humano. O objetivo deste projeto foi criar um modelo estatístico capaz de analisar o histórico e o perfil de um cliente e prever automaticamente se ele será um **Bom Pagador** ou um **Mau Pagador**.

## 🚀 O Grande Diferencial: "From Scratch"
Diferente de projetos convencionais que utilizam bibliotecas como `scikit-learn` para resolver o problema com uma linha de código, este projeto foi construído **do absoluto zero**. Toda a matemática, álgebra linear e probabilidade foram implementadas na mão utilizando apenas **Python e NumPy**.

Isso inclui a construção manual de:
* K-Fold Cross Validation (5 folds).
* Conversão de variáveis categóricas (Label Encoding).
* Matrizes de Distância (Euclidiana e Manhattan).
* Cálculo de Matrizes de Covariância e Vetores de Média.

## 🧠 Algoritmos Implementados
1. **K-Nearest Neighbors (k-NN):** Modelo *lazy learner* baseado no cálculo de distância entre os vizinhos mais próximos.
2. **Naive Bayes (Univariado):** Classificador probabilístico assumindo independência estatística entre as características do cliente.
3. **Bayesiano (Multivariado):** Classificador probabilístico avançado que utiliza a **Matriz de Covariância** para entender a correlação entre os dados (Ex: A relação entre a idade do cliente e o prazo do empréstimo).

## 📊 Resultados e Performance
O modelo Bayesiano Multivariado apresentou o melhor desempenho geral, provando que entender a correlação cruzada entre os dados financeiros do cliente é essencial para prever o risco de crédito.

| Classificador | Acurácia | Precisão | F1-Score | Tempo de Treino | Tempo de Teste |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-vizinhos (Dist. Euclidiana)** | 0.65 ± 0.03 | 0.71 ± 0.03 | 0.77 ± 0.02 | ~ 0.0000s | ~ 0.0122s |
| **K-vizinhos (Dist. Manhattan)** | 0.65 ± 0.03 | 0.72 ± 0.03 | 0.77 ± 0.02 | ~ 0.0000s | ~ 0.0103s |
| **Bayesiano (Univariado)** | 0.68 ± 0.02 | 0.78 ± 0.03 | 0.77 ± 0.02 | ~ 0.0002s | ~ 0.0044s |
| **Bayesiano (Multivariado)** | **0.70 ± 0.04** | **0.78 ± 0.03** | **0.78 ± 0.03** | ~ 0.0008s | **~ 0.0021s** |

*(Nota: Como esperado teoricamente, o K-NN apresentou custo computacional de teste muito superior aos modelos paramétricos bayesianos).*

## 🛠️ Como rodar o projeto
1. Clone este repositório: `git clone [COLE_AQUI_O_LINK_DO_SEU_REPO]`
2. Instale as dependências: `pip install numpy pandas`
3. Execute o script principal: `python main.py`
