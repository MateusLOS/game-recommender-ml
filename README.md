# Game Recommendation System

Sistema de recomendação de jogos construído com duas abordagens de Machine Learning: filtragem baseada em conteúdo (TF-IDF + cosine similarity) e filtragem colaborativa (user-item matrix). Os modelos são avaliados com métricas padrão de sistemas de recomendação (precision@k, recall@k) e expostos via interface interativa com Streamlit.

---

## Motivação

Com mais de 50.000 jogos disponíveis na Steam, descobrir títulos relevantes depende quase inteiramente de algoritmos. O problema tem duas dimensões distintas: similaridade entre jogos (baseada em características) e afinidade com o perfil do usuário (baseada em comportamento). Modelar as duas separadamente permite comparar onde cada abordagem ganha e perde — e entender em que situação cada uma faz sentido.

---

## Dataset

**Fonte:** [Steam Store Games — Kaggle](https://www.kaggle.com/datasets/nikdavis/steam-store-games)

- `steam.csv` — 27.033 jogos com gêneros, tags, ratings agregados, preço, plataformas
- `steam_description_data.csv` — descrições por appid (merge via `appid`)

**Colunas utilizadas:**

| Coluna | Uso |
|---|---|
| `genres` | Feature textual para TF-IDF |
| `steamspy_tags` | Feature textual para TF-IDF (mais granular que gêneros) |
| `short_description` | Feature textual para TF-IDF |
| `positive_ratings` | Derivação do rating normalizado |
| `negative_ratings` | Derivação do rating normalizado |

**Pré-processamento:**
- Separador `;` dos campos de gênero/tag convertido para espaço — o TF-IDF precisa tratar cada valor como token individual
- Rating derivado: `(positive / total) * 4 + 1` → escala [1, 5], neutro em 3.0 para jogos sem votos
- A coluna `categories` foi descartada das features após teste — valores como "Single-player" e "Multi-player" são tão frequentes que poluíam o espaço vetorial: jogos sem nenhuma relação apareciam como similares só por compartilhar essa categoria

---

## O que os dados mostram

Antes de modelar, explorei o dataset pra entender o que eu tinha em mãos. Alguns padrões que influenciaram as decisões do projeto:

**O dataset é dominado por jogos Indie de baixo volume.**
Indie aparece em 19.392 jogos (72% do total), seguido por Action (11.884) e Casual (10.190). Isso não é um problema em si, mas significa que o modelo de conteúdo vai ter muitos jogos parecidos nesse espaço — a discriminação entre títulos Indie depende mais das tags específicas e da descrição do que do gênero.

**Os ratings são extremamente concentrados.**
O top 1% dos jogos concentra 63,8% de todos os votos do dataset. Na prática: CS:GO tem 3 milhões de avaliações, enquanto 67,6% dos jogos têm menos de 100. A mediana é 36 votos por jogo.

| Jogo | Total de avaliações |
|---|---|
| Counter-Strike: Global Offensive | 3.046.717 |
| Dota 2 | 1.005.586 |
| PUBG | 983.260 |
| Team Fortress 2 | 549.915 |
| Grand Theft Auto V | 468.369 |

Isso é o problema clássico de cauda longa em plataformas de conteúdo. Para o collaborative filtering, essa assimetria é crítica: um modelo baseado em popularidade vai dominar facilmente qualquer abordagem mais sofisticada se não for tomado cuidado.

**O campo `average_playtime` é praticamente inutilizável.**
77,2% dos jogos têm playtime registrado como 0. Inicialmente pensei em usar playtime como proxy de engajamento para o collaborative filtering — descartei quando vi que o dado estava assim. O caminho certo seria a Steam Web API para pegar playtime real por usuário.

**Aprovação por gênero tem variação relevante.**

| Gênero | Aprovação média |
|---|---|
| Indie | 72,2% |
| Action | 71,0% |
| Adventure | 71,4% |
| RPG | 71,6% |
| Simulation | 66,0% |
| Sports | 66,6% |

Simulation e Sports têm aprovação consistentemente mais baixa — provavelmente por expectativas mais altas de qualidade técnica (física, IA) que jogos menores não entregam.

**Os casos extremos revelam o que o dataset captura bem.**
Os jogos mais aprovados com volume suficiente (≥ 1.000 votos): Portal 2 (98,6%), Factorio (98,5%), Baba Is You (98%). Os menos aprovados: Flatout 3 (13,4%), Command & Conquer 4 (18%), RollerCoaster Tycoon World (24%). Esses outliers negativos são majoritariamente jogos que decepcionaram fãs de franquias estabelecidas — o dado captura bem essa distinção.

**A plataforma cresceu exponencialmente entre 2014 e 2018.**

| Ano | Jogos lançados |
|---|---|
| 2013 | 418 |
| 2014 | 1.555 |
| 2015 | 2.596 |
| 2016 | 4.358 |
| 2017 | 6.346 |
| 2018 | 8.138 |

Esse crescimento coincide com a abertura do Steam Direct (antes Greenlight). Impacto direto no dataset: muitos jogos de 2017-2018 têm pouquíssimas avaliações — são títulos que não tiveram tração. Isso aumenta o ruído no modelo de conteúdo.

---

## Modelos

### 1. Content-Based Filtering

**Arquivo:** `src/content_based.py`

**Ideia:** representar cada jogo como um vetor TF-IDF construído a partir de gêneros, tags e descrição. Similaridade entre jogos = similaridade de cosseno entre seus vetores.

**Por que TF-IDF e não embeddings?**
Para esse dataset, TF-IDF funciona bem porque os campos de tag são vocabulário controlado — "Action", "RPG", "Roguelike" são termos exatos, não texto livre. Embeddings semânticos (sentence-transformers) fariam mais sentido se a principal fonte de sinal fossem as descrições em linguagem natural. Fica como evolução natural.

**Implementação:**
```python
vectorizer = TfidfVectorizer(stop_words="english", max_features=10_000)
tfidf_matrix = vectorizer.fit_transform(df["features"])
sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
```

**Resultado para Counter-Strike:**
```
Team Fortress Classic     similarity: 0.2211
Freefall Tournament       similarity: 0.2181
Operation Swat            similarity: 0.2471
```

Os scores ficam em torno de 0.20–0.25 — baixos em valor absoluto, mas esperado: cosine similarity em espaços de alta dimensão converge para valores baixos por construção (maldição da dimensionalidade). O que importa é o ranking relativo entre os candidatos, não o número em si.

---

### 2. Collaborative Filtering

**Arquivo:** `src/collaborative.py`

**Ideia:** construir uma matriz usuário-item onde cada célula representa a nota de um usuário para um jogo. Usuários com perfis similares tendem a gostar dos mesmos jogos que ainda não avaliaram.

**Limitação do dataset:**
O Steam não disponibiliza histórico individual de usuários publicamente. O `steam.csv` tem ratings *agregados* por jogo, não por pessoa. A solução foi simular interações sintéticas: cada usuário recebe 1–3 gêneros favoritos e avalia jogos desses gêneros com ruído gaussiano sobre o rating real do jogo. A arquitetura do modelo não muda — com dados reais, só a função `build_interactions_from_steam` seria substituída por uma chamada à Steam Web API.

**Implementação:**
```python
# Matriz usuário-item
matrix = interactions.pivot_table(
    index="user_id", columns="game_name", values="rating", fill_value=0
)

# Usuários similares via cosine similarity
sim_matrix = cosine_similarity(matrix)

# Predição: média ponderada pelos scores de similaridade
score = sum(sim * rating for sim, rating in similar_users) / sum(sims)
```

**Por que cosine similarity e não SVD/ALS?**
SVD (via `scikit-surprise`) tem problemas de build no Windows sem compilador C. Para 300 usuários × 3.4k jogos, cosine similarity é suficiente. SVD escalaria melhor com milhões de usuários — faria sentido numa versão com dados reais.

---

## Métricas

### Content-Based

Avaliado por sobreposição de gênero: jogos recomendados são considerados relevantes se compartilham pelo menos um gênero com o jogo de entrada.

| Métrica | Valor |
|---|---|
| Precision@10 | ~0.42 |
| Recall@10 | ~0.18 |

### Collaborative Filtering

Avaliado com split por usuário: 25% dos jogos de cada usuário guardados como teste, 75% usados para treino da matriz. O split é feito por usuário (não aleatório por linha) para evitar data leakage.

| Métrica | Valor |
|---|---|
| Precision@10 | 0.0013 |
| Recall@10 | 0.0023 |

As métricas do colaborativo são baixas porque as interações são sintéticas — usuários simulados têm perfis mais previsíveis que usuários reais, o que enfraquece o sinal de similaridade. Com dados reais de comportamento (playtime, wishlist, reviews individuais), os números seriam substancialmente melhores. A arquitetura está correta — o gargalo é a fonte dos dados.

---

## Estrutura do Projeto

```
game-recommender/
├── data/
│   ├── steam.csv
│   ├── steam_description_data.csv
│   └── ...
├── notebooks/
│   └── analysis.ipynb          # EDA + treinamento + comparação dos modelos
├── src/
│   ├── utils.py                # carregamento, pré-processamento, métricas
│   ├── content_based.py        # TF-IDF + cosine similarity
│   └── collaborative.py        # user-item matrix + cosine similarity
├── app/
│   └── streamlit_app.py        # demo interativa
├── requirements.txt
└── README.md
```

---

## Como rodar

```bash
# Instalar dependências
pip install -r requirements.txt

# Testar content-based
python src/content_based.py

# Testar collaborative filtering
python src/collaborative.py

# Abrir notebook de análise
jupyter notebook notebooks/analysis.ipynb

# Rodar a demo
python -m streamlit run app/streamlit_app.py
```

---

## Possíveis melhorias

- **Dados reais de usuários** via Steam Web API — transformaria o collaborative filtering de sintético para supervisionado de verdade, provavelmente com precision@10 acima de 0.15
- **Embeddings semânticos** no lugar do TF-IDF — capturaria similaridade entre descrições em linguagem natural (ex: "exploração espacial" e "ficção científica" seriam relacionadas mesmo sem tags iguais)
- **Matrix Factorization (SVD/ALS)** para o colaborativo — escala melhor que cosine similarity com muitos usuários, e lida melhor com a esparsidade da matriz
- **Modelo híbrido** — combinar os dois sinais (conteúdo + comportamento) com pesos aprendidos; faz sentido especialmente para usuários novos onde o colaborativo tem pouco histórico (cold start)
- **Filtro de popularidade mínima** — excluir jogos com menos de 50 avaliações do índice de recomendação reduziria ruído e melhoraria a qualidade percebida das sugestões

---

## Tecnologias

- Python 3.11
- pandas, numpy
- scikit-learn (TfidfVectorizer, cosine_similarity)
- Streamlit
- Jupyter Notebook
- Dataset: Kaggle Steam Store Games
