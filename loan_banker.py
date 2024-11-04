# -*- coding: utf-8 -*-

# importer les packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit # pas ça train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# lire la base de données
df =  pd.read_csv('./train_u6lujuX_CVtuZ9i.csv')
df

# activer: afficher toute la base de données
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df

# desactiver le max de lignes et colonnes
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
df

# voir les valeurs manquantes
""" Des valeurs sont detectées lorsque la taille des valeurs est inferieure a la taille des lignes"""
df.info()

# ou encore on fait comme ça surtout, le nombre de valeurs manquantes dans chaque categories
df.isnull().sum().sort_values(ascending=False)

# afficher les statisques de la base de données
df.describe(include='all')

# afficher les statisques de la base de données | données quantitatives
df.describe()

# afficher les statisques de la base de données | données qualitatives
df.describe(include='O')

"""
les methodes de remplisage des valeurs manquantes sont multiples, le choix adviendra au data specialiste
 mais on peut citer:
 1- supprression si elles ne sont pas trop nombreuses
 2- remplacer par la valeur qui a plus de tendance
 3- remplacer les par les statistiques de base de données (moyenne, mediane, mode).
 4- remplacer les par des valeurs aleatoires.
 5- remplacer par une valeur constante.
 6- remplacer par la methode de KNN
 7- remplacer par la methode de regression.
"""

# ici on remplacera par le voisin
# renseigner les manquantes

# d'abord, separer la base de données en 2 (categorie, numerique)
cat_data = [] # categorie
num_data = [] # numerique


for i, c in enumerate(df.dtypes):
  if c == object:
    cat_data.append(df.iloc[:, i])
  else:
    num_data.append(df.iloc[:, i])

cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()

#database numerique
num_data

# database categorique
cat_data

# Pour les variables categoriques on va remplacer les valeurs manquantes par les valeurs qui se repete de plus
print(cat_data["Education"].value_counts()) # donne une liste des valeurs en categorie des valeurs par ordre decroissant
cat_data = cat_data.apply(lambda x: x.fillna(x.value_counts().index[0])) # remplacer les valeurs manquantes par les valeurs qui se repete de plus
cat_data.isnull().sum().any() # verifie si il reste des valeurs manquantes

# ou comme ça
cat_data.info() # tout est correct

# Pour les variables numeriques, on va remplacer les valeurs manquantes par la valeur precedante de la meme colonne
num_data.fillna(method='bfill', inplace =True) # des methodes de pandas bfill, ffill
num_data.isnull().sum().any() # verifie si il reste des valeurs manquantes

# ou encore
num_data.info() # tout est correct

# afficher la data
num_data

# transformer des valeurs de certaines colonnes pour pouvoir analyser la base de données

# transformer la colonne target
target_value = {"Y":1, "N":0}
target = df["Loan_Status"]
cat_data.drop("Loan_Status", axis=1, inplace=True)
target = target.map(target_value)
target

# maintenant transformer les colonnes categorie automatiquement avec sklearn
# remplacer les valeurs catégoriques par des numeriques 1,2,3,..
le = LabelEncoder()
for i in cat_data:
  cat_data[i] = le.fit_transform(cat_data[i])
cat_data

# on supprime loan_id
cat_data.drop("Loan_ID", axis=1, inplace=True)

# concatener cat_data et num_data et specifier la colonne target

X = pd.concat([cat_data, num_data], axis=1)
y = target
X

"""
Cette partie est tres importante et tres mesquins, il s'agit de l'ingénierie du caractere.
"""

"""### Analyse exploratoire des données (EDA)"""

# On va commencer par la variable target
target.value_counts()

plt.figure(figsize=(8, 6))
sns.countplot(x=target)
yes = target.value_counts()[1]/len(target)*100
no =  target.value_counts()[0]/len(target)*100
plt.text(0, yes+1, '{:.2f}%'.format(yes), ha='center')
plt.text(1, no+1, '{:.2f}%'.format(no), ha='center')
plt.title('Target distribution')
print('Le pourcentage des credits accordés est: {:.2f}%'.format(yes))
print('Le pourcentagage des credits non accordés est: {:.2f}%'.format(no))
plt.show()

# base de données
df = pd.concat([X, y], axis=1)

# credit history
grid = sns.FacetGrid(df, col='Loan_Status', aspect=1.6)
grid.map(sns.countplot, 'Credit_History')

# Sexe
grid = sns.FacetGrid(df, col='Loan_Status', aspect=1.6)
grid.map(sns.countplot, 'Gender')

# Status Matrimoniale
grid = sns.FacetGrid(df, col='Loan_Status', aspect=1.6)
grid.map(sns.countplot, 'Married')

# Education (etude)
grid = sns.FacetGrid(df, col='Loan_Status', aspect=1.6)
grid.map(sns.countplot, 'Education')

# revenu du demandeur
plt.scatter(df['ApplicantIncome'], df['Loan_Status'])
plt.xlabel('ApplicantIncome')
plt.ylabel('Loan_Status')
plt.show()
print("Pas trop d'impact entre le revenu et la capacité du pret")

# revenu du demandeur
plt.scatter(df['CoapplicantIncome'], df['Loan_Status'])
plt.xlabel('CoapplicantIncome')
plt.ylabel('Loan_Status')
plt.show()
print("Pas trop d'impact entre le revenu du conjoint et la capacité du pret")

df.groupby('Loan_Status').median()

# le but de l'analyse exploratoire est de trouver
# une variable independante à la variable dependante target. pour voir quelle est
# l'impact de cette variable sur la prise de décision
# c'est pourquoi on regarde dans le visuel, l'impact de chaque element.

"""### realisation du model"""

# diviser la base données en une base de données test et d'entrainemen

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X, y):
  X_train, X_test = X.iloc[train], X.iloc[test]
  y_train, y_test = y.iloc[train], y.iloc[test]

print("X_train taille: ", X_train.shape)
print("X_test taille: ", X_test.shape)
print("y_train taille: ", y_train.shape)
print("y_test taille: ", y_test.shape)

# On va appliquer 3 algorithmes, Regression logistique, KNN, DecisionTree

models = {
    'LogisticRegression':LogisticRegression(random_state=42),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=1, random_state=42)
}

# la fonction de precision

def accuracy(y_true, y_pred, retu=False):
    acc = accuracy_score(y_true, y_pred)
    if retu:
        return acc
    else:
        print("La précision du modèle: {:.2f}%".format(acc*100))

# c'est la fonction d'application des models

def train_test_eval(models, X_train, y_train, X_test, y_test):
    for name, model in models.items():
        print(f"Entrainement du modèle: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy(y_test, y_pred)
        print('-'*50)

train_test_eval(models, X_train, y_train, X_test, y_test)

# on va créer une nouvelle base de données

X_2 = X[["Credit_History", "Gender", "Married", "CoapplicantIncome","ApplicantIncome"]]

# diviser la base données en une base de données test et d'entrainemen

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X_2, y):
  X_train, X_test = X_2.iloc[train], X_2.iloc[test]
  y_train, y_test = y.iloc[train], y.iloc[test]

print("X_train taille: ", X_train.shape)
print("X_test taille: ", X_test.shape)
print("y_train taille: ", y_train.shape)
print("y_test taille: ", y_test.shape)

# on va encore diminuer le nombre de variables pour voir si le model sera amelioré

X_2 = X[["Credit_History", "Married", "CoapplicantIncome"]]

# diviser la base données en une base de données test et d'entrainemen

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sss.split(X_2, y):
  X_train, X_test = X_2.iloc[train], X_2.iloc[test]
  y_train, y_test = y.iloc[train], y.iloc[test]

print("X_train taille: ", X_train.shape)
print("X_test taille: ", X_test.shape)
print("y_train taille: ", y_train.shape)
print("y_test taille: ", y_test.shape)

# on applique enore sur le model

train_test_eval (models, X_train, y_train, X_test, y_test)

"""### deploiement"""

# Appliquer la regression logistique sur notre base ded données
model = LogisticRegression(random_state=42)
model.fit(X_2, y)

# On enregistre le model

pickle.dump(model, open('model.pkl', 'wb'))