import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(path + '/airlines_flights_data.csv')
df
df.drop(columns='index', inplace=True)
df.head()
df.describe()
df.columns

# Menampilkan visualisasi outlier pada kolom numerik
plt.figure(figsize=(10,8))
df.boxplot(column=['days_left','duration'])
plt.title('Outlier Detection Across Key Variables')
plt.xlabel('Variabel Data')
plt.ylabel('Nilai')
plt.grid(True)
plt.show()
