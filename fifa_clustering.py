import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def main():
    df = data_preprocessing()
    df_norm = standardization(df)
    df_new = dimensional_reduction(df_norm)
    
def data_preprocessing():
    #Import the dataset
    df = pd.read_csv('players_21.csv')
    #Take the necessary features out of all the features
    df = df[['short_name','age', 'height_cm', 'weight_kg', 'overall', 'potential','international_reputation', 'weak_foot','skill_moves', 'attacking_crossing', 'attacking_finishing',
    'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power',
    'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties','mentality_composure', 'defending_standing_tackle',
    'defending_sliding_tackle', 'goalkeeping_diving','goalkeeping_handling', 'goalkeeping_kicking','goalkeeping_positioning', 'goalkeeping_reflexes']]

    #Filter only the players with rating above 80
    df = df[df['overall'] > 80]
    #Fill the missing data
    df = df.fillna(df.mean())
    return df

def standardization(df):
    #Applying standardization to change the value into 0-1
    x = df.iloc[:,1:] #Numpy array
    sc = MinMaxScaler()
    X_norm = sc.fit_transform(x)
    df_norm = pd.DataFrame(X_norm)
    return df_norm

def dimensional_reduction(df):
    #Applying the dimensionality reduction to reduce the dimensions to 3
    pca = PCA(n_components = 3)
    df_new = pd.DataFrame(pca.fit_transform(df))
    return df_new

def clustering():
    #Using KMeans Algorithm for machine learning modelling
    kmeans = KMeans(n_clusters = 4) #specify number of clusters
    kmeans = kmeans.fit(df_new) #fit the dataset
    labels = kmeans.predict(df_new) #predict which players belong to a certain cluster
    cluster = kmeans.labels_.tolist() #obtain the cluster number for each players

    #Displaying the output of predicted cluster
    df_new.insert(loc = 0, column = 'Name', value = df['short_name'])
    df_new.insert(loc = 1, column = 'Cluster',value = cluster)
    df_new.columns = ['Name', 'Cluster', 'Feature 1', 'Feature 2', 'Feature 3']

