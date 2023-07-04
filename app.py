import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


df = pd.read_csv('go_track_trackss.csv')

df.rename(index=str, columns={
    'speed': 'kecepatan',
    'distance': 'jarak'
}, inplace=True)

X = df.drop(['rating_bus', 'rating_weather', 'rating', 'linha'], axis=1)

st.header("Isi dataset")
st.write(X)

# Menampilkan panah elbow

clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(10, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')

# panah elbow
ax.annotate('possible elbow point', xy=(3, 14000), xytext=(3, 50000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))


plt.show()
st.header('Data Set Menggunakan Elbow')
st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih Jumlah Clusters :", 2, 10, 3, 1)


def k_means(n_clust):
    kmeans = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmeans.labels_

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=X['kecepatan'], y=X['jarak'], hue=X['Labels'], markers=True, size=X['Labels'],
                    palette=sns.color_palette('hls', len(X['Labels'].unique())))

    for label in X['Labels']:
        plt.annotate(label,
                     (X[X['Labels'] == label]['kecepatan'].mean(),
                      X[X['Labels'] == label]['jarak'].mean()),
                     fontsize=20, weight='bold',
                     color='black')

    st.header('Cluster Plot')
    st.pyplot()
    st.write(X)


k_means(clust)
