# Nested Monte Carlo for Log-Variance-Gamma Model

Ce projet utilise la simulation Monte Carlo sur GPU (CUDA) pour calculer le prix d'options call européennes dans le modèle Variance Gamma. Quatre réseaux de neurones MLP sont ensuite entraînés pour approximer ces prix à partir des paramètres `T`, `K`, `kappa`, `theta` et `sigma`.

## 1. Organisation du projet

```text
├── data/             # Génération et stockage des données CSV
├── monte_carlo/      # Simulation Monte Carlo et benchmarks CUDA
├── neural_network/   # Préparation des données, modèles et entraînement
├── analyse/          # Évaluation des modèles et propriétés financières
├── outputs/          # Modèles entraînés et résultats
├── requirements.txt
└── README.md
```

## 2. Installation

Prérequis : Python 3 (3.11 recommandé) et, pour la partie Monte Carlo, une carte graphique NVIDIA compatible CUDA avec le compilateur `nvcc`.

Depuis la racine du projet, installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

L'entraînement et les analyses Python peuvent être exécutés sur CPU ou sur GPU avec une installation de PyTorch compatible CUDA. Le périphérique est choisi automatiquement.

## 3. Génération des données

Si les fichiers `data/entree.csv` et `data/sorties.csv` sont déjà disponibles, passer directement à l'étape 4.

Générer les 100 000 jeux de paramètres :

```bash
python data/generer_entrees.py data/entree.csv
```

Compiler et exécuter la simulation Monte Carlo :

```bash
cd monte_carlo
nvcc -O3 vg_nested_mc.cu -o vg_nested_mc
./vg_nested_mc
cd ..
```

Les prix et leurs intervalles de confiance à 95 % sont enregistrés dans `data/sorties.csv`.

## 4. Entraînement des réseaux de neurones

Depuis la racine du projet :

```bash
python -m neural_network.train
```

Les données sont réparties en trois ensembles : entraînement (70 %), validation (15 %) et test (15 %), avec des proportions approximatives.

Quatre architectures MLP sont comparées : `128-128-128-128`, `128-128-64`, `128-128-64-32` et `128-128-64` avec pénalités financières. Le meilleur modèle est sélectionné selon la RMSE sur l'ensemble de validation.

Les modèles, les paramètres de normalisation et les historiques sont sauvegardés automatiquement dans `outputs/`.

## 5. Analyse des résultats

Si les modèles entraînés, les scalers correspondants et le fichier de test sont déjà présents dans `outputs/`, ces commandes peuvent être exécutées directement après l'installation.

```bash
python -m analyse.evaluate
python -m analyse.analyze_shape
```

- `evaluate.py` : comparaison des prix prédits aux prix Monte Carlo (MSE, RMSE, MAE et R²).
- `analyze_shape.py` : vérification numérique de la monotonie en maturité et en strike, ainsi que de la convexité en strike, sur plusieurs coupes du domaine.

## 6. Benchmark Monte Carlo (facultatif)

```bash
cd monte_carlo
nvcc -O3 bench_q2.cu -o bench_q2
./bench_q2 all
cd ..
```

Ce programme compare différentes configurations CUDA, étudie la convergence Monte Carlo et compare un prix GPU à une estimation CPU indépendante. Les mesures sont enregistrées dans `monte_carlo/benchmarks_q2.csv` et la comparaison CPU/GPU est affichée dans le terminal.

## 7. Résultats

Les résultats Python sont organisés dans `outputs/` :

| Dossier | Contenu |
| --- | --- |
| `models/` | Poids des réseaux entraînés (`.pth`) |
| `scalers/` | Paramètres de normalisation communs (`.joblib`) |
| `histories/` | Pertes au cours de l'entraînement |
| `figures/` | Courbes de validation, comparaison des prix et propriétés financières |
| `results/` | Métriques, prédictions, diagnostics, test set et modèle sélectionné |

Les principaux résultats numériques se trouvent dans `benchmark_validation.csv`, `benchmark_test.csv` et `shape_diagnostics_all_models.csv`, dans `outputs/results/`.

Les scripts Python doivent être lancés depuis la racine du projet. Les programmes CUDA doivent être exécutés depuis `monte_carlo/` comme indiqué ci-dessus. Une nouvelle exécution écrase les fichiers de résultats correspondants.
