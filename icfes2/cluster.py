import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

def preprocess(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    X = df[cols].fillna(df[cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, cols

def run_kmeans(df, n_clusters=3, cols=None):
    X_scaled, scaler, cols = preprocess(df, cols)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    df["cluster_kmeans"] = labels
    return df, model

def run_dbscan(df, eps=0.5, min_samples=5, cols=None):
    X_scaled, scaler, cols = preprocess(df, cols)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    df["cluster_dbscan"] = labels
    return df, model

def run_hierarchical(df, n_clusters=3, cols=None):
    X_scaled, scaler, cols = preprocess(df, cols)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    df["cluster_hierarchical"] = labels
    return df, model
