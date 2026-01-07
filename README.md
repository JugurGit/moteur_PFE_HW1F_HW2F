# Mémoire — Hull-White 1F / 2F (G2++) • Calibration • PFE (Streamlit)

Application Streamlit multi-pages + librairie Python qui permet de :
- **Charger une courbe** (discount factors) depuis un template Excel
- **Calibrer Hull–White 1F** : paramètres (a, σ) sur swaptions (convention forward premium)
- **Calibrer Hull–White 2F (G2++)** : profile calibration (a, b, ρ) + inner (σ, η)
- **Comparer Market vs Model** (prices + implied normal vols Bachelier)
- **Calculer PFE / EPE d’un swap vanilla** via Monte Carlo (1F et 2F)
- **Portfolio tracking mode** : logique d’historisation de runs (via SQLite, initialisée au démarrage)
- **Documentation** : navigation dans le code depuis l’UI

> ⚠️ Projet à but illustratif : les conventions, paramètres par défaut et templates “demo” ne constituent pas une implémentation production/réglementaire.

👉 Démo en ligne : **https://boudarene-moteurpfe.streamlit.app/**
---

## 1) Prérequis

- **Python 3.10+** (recommandé)

### 2) Récupérer le projet
#### Option A — via Git
```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
```

#### Option B — via ZIP
- Télécharger le ZIP depuis GitHub
- Le dézippez
- Ouvrir un terminal dans le dossier du projet

### 3) Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4) Lancer l’application Streamlit
```bash
streamlit run streamlit_app/app.py
```
Streamlit va afficher une URL du type :
- Local: http://localhost:8501

### 5) Utilisation rapide
#### Overview
Résumé des fonctionnalités principales de l’application.

#### Calibration HW1F (page 2)
- **Charger les données** :
  - par défaut, des données de calibration de **swaption** sont déjà chargées 
  - uploader un template **swaption** `.xlsx`, **ou**
  - utiliser celui fourni dans le repo : `Calibration_Templates/`
- **Lancer la calibration** des paramètres **(a, σ)**, puis consulter :
  - les **logs** du calibrator
  - le tableau **Market vs Model**
  - les **plots** (prix & volatilités)

#### Calibration HW2F (page 3)
- Même workflow que HW1F, avec une calibration en 2 niveaux :
  - **outer grid** : **(a, b, ρ)**
  - **inner grid** : **(σ, η)**
- Suivre la progression via :
  - la **barre de progression**
  - les **logs**
    
#### PFE Swap (page 4)
- **Configurer le calcul** :
  - choix du modèle (**HW1F** / **HW2F**)
  - quantile, grille, notional, schedule
- **Lancer le calcul** **PFE/EPE** et visualiser le **profil** (courbes / métriques)

#### Portfolio Tracking
- Activer / désactiver le mode via le toggle **📌 Portfolio tracking mode** dans la sidebar
- Sauvegarder et consulter des **runs** (selon l’implémentation de la page tracking)

### 6) Lancer le moteur en ligne de commande via un notebook

Le script main.py exécute un run “console” :

**Mode démo**
```bash
python test.ipynb
```
