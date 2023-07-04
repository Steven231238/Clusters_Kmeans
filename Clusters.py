import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Membaca file csv menggunakan pandas
driver = pd.read_csv("go_track_trackss.csv")
print(driver)

# Memilih atribut yang diperlukan
driver_x = driver.drop(["linha", "car_or_bus", "rating_weather", "rating", "time", "id", "id_android"], axis=1)
print(driver_x)

# Mengubah data frame menjadi array numpy dan mengganti nilai NaN dengan 0 dan infinity dengan nilai terbesar
x_array = np.array(driver_x)
x_array = np.nan_to_num(x_array)

# Normalisasi data menggunakan MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)

# Memasukkan array yang sudah dinormalisasi ke dalam algoritma KMeans
a = int(input("Masukkan jumlah Cluster yang Anda inginkan: "))
kmeans = KMeans(n_clusters=a, n_init=10, random_state=123)
kmeans.fit(x_scaled)

# Menampilkan data kluster di konsol
for i in range(len(driver)):
    print("id record:", driver.values[i, 0], ", id android:", driver.values[i, 1], ", Cluster:", kmeans.labels_[i])
    print("----------------------------------------------------------------------------------")

# Menyimpan plot klustering sebagai file gambar
matplotlib.use('Agg')
driver["kluster"] = kmeans.labels_
output = plt.scatter(x_scaled[:, 0], x_scaled[:, 1], s=100, c=driver.kluster, marker="o", alpha=1)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=1, marker="s")
plt.title("Hasil klustering K-Means")
plt.colorbar(output)
plt.savefig("plot.png")  # Menyimpan plot sebagai file gambar
plt.close()  # Menutup plot yang disimpan
