import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.datasets

# Fonction de chargement des données
def load_planar_dataset():
    np.random.seed(1)
    m = 400  
    N = int(m / 2)  
    D = 2  
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4  

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X, Y

# Fonction pour afficher la frontière de décision
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral, edgecolors='k')
    st.pyplot(plt)

# Interface utilisateur Streamlit
st.title("Classification des données planaires")

# Barre latérale pour le menu
menu = st.sidebar.radio("Menu", ("Contexte", "Modèle de Régression Logistique", "Modèle Réseau de Neurones", "Tableaux comparatifs"))

# Affichage du contenu en fonction du choix dans la barre latérale
if menu == "Contexte":
    st.subheader("Contexte")
    st.write("""
    L'objectif est de **construire un modèle de classification binaire** en utilisant un réseau de neurones avec une couche cachée. 
    On commence par entraîner un modèle de régression logistique simple pour observer ses limites. 
    Ensuite, on implémente un réseau de neurones plus avancé pour améliorer les performances sur un jeu de données non linéairement séparable.
    """)
    X, Y = load_planar_dataset()
    Y = Y.reshape(-1, 1)
    
    st.subheader("Visualisation des données")
    st.write("""
    Les données sont disposées en forme de fleur, où chaque classe est distribuée de manière non linéaire.
    Pour la structure on est parti sur 400 points répartis en deux classes représentées par la couleur rouge(Classe 0) et la couleur bleu(Classe 1)
    
    """)
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.Spectral, edgecolors='k')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Données de classification")
    st.pyplot(fig)
    st.write("""
    ### Observation sur les données : 
    Les points sont répartis en forme de fleur avec deux classes distinctes (rouge et bleu), ce qui confirme que la séparation des classes n'est pas linéaire. 
    """)

elif menu == "Modèle de Régression Logistique":
    st.subheader("Modèle de Régression Logistique")
    X, Y = load_planar_dataset()
    Y = Y.reshape(-1, 1)
    
    # Entraînement du modèle de régression logistique
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, Y.ravel())
    accuracy = clf.score(X, Y) * 100
    st.write(f"Précision du modèle : {accuracy:.2f}%")
    
    # Affichage de la frontière de décision
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    
    # Observation
    st.subheader("Observation")
    st.write("""
    La régression logistique échoue à bien classifier les données car la séparation des classes est non linéaire.
    La précision obtenue est très faible (≈ 47%), ce qui explique que de nombreux points rouges se trouvent dans la zone bleue
    et vice versa. Le modèle ne capte pas la structure en forme de fleur des données.
    """)


elif menu == "Modèle Réseau de Neurones":
    st.subheader("Modèle Réseau de Neurones")
    X, Y = load_planar_dataset()
    Y = Y.reshape(-1, 1)
    st.subheader("Initialisation du réseau de neurones")
    st.write("""
    Pour ce modèle, nous utilisons un réseau de neurones à une seule couche cachée. 
    Les paramètres du modèle sont initialisés de manière aléatoire avec des valeurs proches de zéro.

    Voici les étapes d'initialisation des paramètres :
    - **n_x** = X.shape[1] soit 2 : Taille de la couche d'entrée (le nombre de caractéristiques)
    - **n_h** = 4  : Nombre de neurones dans la couche cachée 
    - **n_y** = 1  : Taille de la couche de sortie (1 vu qu'il s'agit d'un problème de classification binaire)

    Le modèle calcule ensuite les sorties à partir de ces paramètres et fait des prédictions sur les données.

    """)

    st.subheader("Optimisation du Coût dans le modèle")
    st.write("""
    **Le coût** : est une fonction utilisée pour mesurer l'écart entre les prédictions du modèle et les valeurs réelles.
    Plus il est faible, mieux le modèle apprend. Une fois que le coût atteint un minimum, cela indique que le modèle a 
    appris à faire des prédictions qui se rapprochent le plus possible des valeurs réelles
    """)
    st.subheader("Évolution du coût par itération")
    st.write("""
    Le tableau ci après montre l'évolution du cout apres chaque itération.
    """)
    cost_iterations = {
        "Itération": [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000],
        "Coût": [0.6930480201239823, 0.2880832935690184, 0.25438549407324534, 0.23386415038952202,
                0.2267924874485401, 0.22264427549299018, 0.21973140404281322, 0.21750365405131297,
                0.21960468183115134, 0.21859799830655824]
    }

    # Création du tableau sous forme de DataFrame
    df_cost = pd.DataFrame(cost_iterations)

    # Affichage du tableau
    st.write(df_cost)

    st.subheader("Observation")
    st.write("""
    Le **coût** diminue rapidement au début de l'entraînement. Après environ 7000 à 9000 itérations, 
    le coût semble se stabiliser, ce qui indique que le modèle a atteint une **convergence optimale**. 
    À ce stade, le modèle est capable de faire des prédictions de manière relativement stable et efficace.
    Pour entraîner notre modèle, nous avons choisi un nombre d'itérations de **10 000**.
    """)

    # Fonction du réseau de neurones (comme dans ton code précédent)
    def initialize_parameters(n_x, n_h, n_y):
        np.random.seed(2)
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(X, parameters):
        W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
        Z1 = np.dot(W1, X.T) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def compute_cost(A2, Y):
        m = Y.shape[0]
        return -np.sum(Y * np.log(A2.T) + (1 - Y) * np.log(1 - A2.T)) / m

    def backward_propagation(parameters, cache, X, Y):
        m = X.shape[0]
        W1, W2 = parameters["W1"], parameters["W2"]
        A1, A2 = cache["A1"], cache["A2"]
        dZ2 = A2 - Y.T
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = np.dot(dZ1, X) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_parameters(parameters, grads, learning_rate):
        parameters["W1"] -= learning_rate * grads["dW1"]
        parameters["b1"] -= learning_rate * grads["db1"]
        parameters["W2"] -= learning_rate * grads["dW2"]
        parameters["b2"] -= learning_rate * grads["db2"]
        return parameters

    def train_nn(X, Y, n_h, num_iterations=10000, learning_rate=1.2):
        n_x, n_y = X.shape[1], 1
        parameters = initialize_parameters(n_x, n_h, n_y)
        for i in range(num_iterations):
            A2, cache = forward_propagation(X, parameters)
            cost = compute_cost(A2, Y)
            grads = backward_propagation(parameters, cache, X, Y)
            parameters = update_parameters(parameters, grads, learning_rate)
        return parameters

    def predict_nn(parameters, X):
        A2, _ = forward_propagation(X, parameters)
        return (A2 > 0.5).astype(int).flatten()

    
    parameters_static = train_nn(X, Y, n_h=4)
    accuracy_static = np.mean(predict_nn(parameters_static, X) == Y.ravel()) * 100
    st.subheader("Graphe avec 4 neurones dans la couche cachée")
    st.write(f"Précision du modèle avec 4 neurones : {accuracy_static:.2f}%")
    
    plot_decision_boundary(lambda x: predict_nn(parameters_static, x), X, Y)
    # Sélection du nombre de neurones pour le graphe dynamique
    n_h_dynamic = st.selectbox("Sélectionner le nombre de neurones dans la couche cachée", [1, 2, 3, 4, 5, 20, 50])
    
    # Entraînement du modèle avec le nombre dynamique de neurones
    parameters_dynamic = train_nn(X, Y, n_h=n_h_dynamic)
    accuracy_dynamic = np.mean(predict_nn(parameters_dynamic, X) == Y.ravel()) * 100
    st.write(f"Précision du modèle avec {n_h_dynamic} neurones : {accuracy_dynamic:.2f}%")
    
    # Affichage de la frontière de décision pour le graphe dynamique
    st.subheader(f"Graphe avec {n_h_dynamic} neurones dans la couche cachée")
    plot_decision_boundary(lambda x: predict_nn(parameters_dynamic, x), X, Y)


elif menu == "Tableaux comparatifs":
    st.subheader("Tableau récapitulatif des résultats sur les tailles des neurones")

    tableau_comparatif1 = [
    {"Nbre de neurones": 1, "Frontière": "Séparation linéaire", "Performance": "Mauvaise", "Commentaire": "Modèle trop simple, ne capture pas la structure en spirale. Performances proches de la régression logistique (~47%)."},
    {"Nbre de neurones": 2, "Frontière": "Frontière légèrement incurvée", "Performance": "Moyenne", "Commentaire": "Légère amélioration mais encore trop de mauvaises classifications."},
    {"Nbre de neurones": 3, "Frontière": "Forme en spirale commence à apparaître", "Performance": "Correcte", "Commentaire": "Commence à mieux séparer les classes, mais encore des erreurs."},
    {"Nbre de neurones": 4, "Frontière": "Bonne séparation des classes", "Performance": "Bon", "Commentaire": "Meilleur compromis entre performance et complexité. Précision améliorée."},
    {"Nbre de neurones": 5, "Frontière": "Frontière plus affinée", "Performance": "Très bon", "Commentaire": "Modèle optimal, bien équilibré. Capture bien la structure sans sur-apprentissage."},
    {"Nbre de neurones": 20, "Frontière": "Séparation très flexible", "Performance": "Risque de sur-apprentissage", "Commentaire": "Modèle très précis mais risque de mémoriser trop les données d'entraînement."},
    {"Nbre de neurones": 50, "Frontière": "Séparation trop complexe", "Performance": "Sur-apprentissage", "Commentaire": "La frontière devient trop irrégulière, ce qui signifie un sur-ajustement aux données. Mauvaise généralisation sur de nouvelles données."}
]

    # Créer un DataFrame pour le tableau
    df_comparatif = pd.DataFrame(tableau_comparatif1)

    # Afficher le tableau avec Streamlit
    st.dataframe(df_comparatif)

    # Affichage du tableau des performances des modèles
    st.subheader("Comparaison des modèles")
    comparison_data = {
        "Critères": [
            "Type de Modèle", "Performance", "Capacité à capturer les non-linéarités", 
            "Adaptation au dataset", "Complexité du Modèle", "Risques", "Meilleure utilisation"
        ],
        "Régression Logistique": [
            "Linéaire", "47% (Faible)", "Non", "Mauvaise (ne sépare pas bien la spirale)", 
            "Simple et rapide", "Aucun sur-apprentissage", "Données séparables linéairement"
        ],
        "Réseau de Neurones": [
            "Non linéaire (1 couche cachée)", "90.75% (Élevée)", "Oui (grâce à la couche cachée)", 
            "Excellente (capture la structure)", "Plus complexe, nécessite plus de calculs", 
            "Possible sur-apprentissage si n_h trop grand", "Données avec des motifs non linéaires"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)