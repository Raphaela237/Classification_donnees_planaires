# Classification de Données Planaires avec Réseau de Neurones

Ce projet illustre l'utilisation d'un réseau de neurones pour classifier un jeu de données synthétique représentant une structure en spirale. L'objectif est de comparer la performance d'un modèle de régression logistique avec celle d'un réseau de neurones à une couche cachée.

## 📌 Objectifs
- Générer un jeu de données synthétique complexe non linéaire.
- Implémenter un réseau de neurones et observer l'évolution des performances en fonction du nombre de neurones cachés.
- Comparer les résultats avec une régression logistique.
- Visualiser les frontières de décision des modèles.

## 📊 Jeu de Données
Les données sont générées de manière artificielle sous la forme de deux classes en spirale. Chaque classe est distribuée de manière non linéaire. Pour la structure on est parti sur 400 points répartis en deux classes représentées par la couleur rouge(Classe 0) et la couleur bleu(Classe 1)
    

## 🛠️ Technologies utilisées
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

## 🚀 Lancement de l'application
```bash
pip install -r requirements.txt
streamlit run app_ml.py
```

## 📈 Comparaison des modèles
Un tableau comparatif met en évidence les performances des modèles en fonction du nombre de neurones dans la couche cachée. Le réseau de neurones dépasse largement la régression logistique pour capturer la structure des données.

## 📸 Visualisation
L'application permet d'afficher les données, la classification des modèles et l'évolution des frontières de décision en fonction du nombre de neurones cachés.

