import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

# Set data

GNB5 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [2.146,2.085,3.500,5.549,5.963,4.159,4.598],
    'Precision': [1.829,1.756,3.220,6.256,5.866,4.720,4.354],
    'Recall': [4.207,5.159,4.902,2.134,3.744,3.329,4.524],
    'Gmean': [2.341,2.695,4.183,4.695,5.890,3.622,4.573],
    'BAC': [2.317,2.634,3.963,4.720,5.976,3.671,4.720],
})

GNB15 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [2.049,2.037,3.646,5.915,5.817,4.146,4.390],
    'Precision': [1.902,1.683,3.366,6.573,5.866,4.512,4.098],
    'Recall': [4.073,5.012,4.439,2.585,3.549,3.683,4.659],
    'Gmean': [2.268,2.500,4.378,4.768,5.524,3.878,4.683],
    'BAC': [2.195,2.476,4.183,4.890,5.573,3.951,4.732],
})

GNB30 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [2.000,2.171,3.878,5.890,5.695,3.976,4.390],
    'Precision': [1.927,1.707,3.415,6.512,5.878,4.451,4.110],
    'Recall': [4.037,5.378,4.780,2.622,3.220,3.598,4.366],
    'Gmean': [2.220,2.659,4.854,4.598,5.012,4.000,4.659],
    'BAC': [2.171,2.634,4.659,4.817,5.110,3.976,4.634],
})

GNB50 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [2.049,2.232,3.793,5.939,5.549,4.171,4.268],
    'Precision': [1.902,1.756,3.585,6.476,5.841,4.463,3.976],
    'Recall': [4.073,5.378,4.634,2.707,3.134,3.866,4.207],
    'Gmean': [2.341,2.646,4.841,4.671,4.939,4.256,4.305],
    'BAC': [2.195,2.695,4.573,4.805,5.000,4.256,4.476],
})

GNB5s = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [2.341,2.280,4.159,5.634,6.098,3.878,3.610],
    'Precision': [2.244,2.098,3.902,6.341,6.098,3.976,3.341],
    'Recall': [4.037,4.890,4.427,1.939,3.305,4.183,5.220],
    'Gmean': [2.341,2.793,4.622,4.829,5.976,3.610,3.829],
    'BAC': [2.341,2.634,4.427,4.829,6.061,3.610,4.098],
})

GNB15s = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [2.463,2.500,4.476,6.000,6.024,3.500,3.037],
    'Precision': [2.390,2.146,4.390,6.646,6.110,3.463,2.854],
    'Recall': [3.902,4.780,3.768,2.366,3.085,4.732,5.366],
    'Gmean': [2.390,2.939,4.915,5.098,5.780,3.451,3.427],
    'BAC': [2.366,2.866,4.646,5.195,5.805,3.549,3.573],
})

GNB30s = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [2.537,2.780,4.756,5.988,5.841,3.354,2.744],
    'Precision': [2.561,2.366,4.463,6.561,6.049,3.293,2.707],
    'Recall': [3.805,4.915,4.012,2.171,2.732,4.890,5.476],
    'Gmean': [2.488,3.195,5.463,4.915,5.305,3.451,3.183],
    'BAC': [2.415,3.098,5.268,5.098,5.488,3.329,3.305],
})

GNB50s = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [2.659,2.890,4.768,6.085,5.720,3.244,2.634],
    'Precision': [2.561,2.537,4.537,6.561,6.073,3.098,2.634],
    'Recall': [3.744,4.878,3.732,2.293,2.683,5.183,5.488],
    'Gmean': [2.756,3.134,5.427,4.963,5.280,3.415,3.024],
    'BAC': [2.585,3.134,5.207,5.207,5.354,3.415,3.098],
})

CART5 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [2.683,2.841,2.988,5.329,5.561,4.256,4.341],
    'Precision': [2.634,3.976,4.195,5.695,5.134,3.195,3.171],
    'Recall': [3.293,2.622,2.695,3.890,4.463,5.366,5.671],
    'Gmean': [3.098,2.671,2.817,4.061,4.634,5.232,5.488],
    'BAC': [3.098,2.585,2.732,4.280,4.829,5.085,5.390],
})

CART15 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [2.817,3.256,3.732,5.195,5.463,3.463,4.073],
    'Precision': [3.012,5.098,5.378,4.683,4.659,2.610,2.561],
    'Recall': [2.854,2.598,2.963,3.756,4.390,5.280,6.159],
    'Gmean': [2.744,2.622,3.098,4.049,4.585,4.902,6.000],
    'BAC': [2.622,2.707,3.183,4.098,4.671,4.854,5.866],
})

CART30 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [2.902,3.402,3.963,5.146,5.415,3.463,3.707],
    'Precision': [3.171,5.707,5.537,4.366,4.463,2.390,2.366],
    'Recall': [2.671,2.415,3.232,3.744,4.390,5.402,6.146],
    'Gmean': [2.585,2.500,3.256,4.049,4.805,5.000,5.805],
    'BAC': [2.537,2.598,3.439,4.110,4.805,4.829,5.683],
})

CART50 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [3.000,3.451,4.061,5.220,5.415,3.366,3.488],
    'Precision': [3.317,5.720,5.622,4.439,4.293,2.244,2.366],
    'Recall': [2.829,2.354,3.134,3.707,4.317,5.488,6.171],
    'Gmean': [2.634,2.476,3.207,4.122,4.707,5.000,5.854],
    'BAC': [2.561,2.671,3.280,4.244,4.854,4.756,5.634],
})

kNN5s = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [3.902,4.963,4.134,4.780,4.878,2.878,2.463],
    'Precision': [5.354,4.695,3.854,5.207,4.293,2.732,1.866],
    'Recall': [1.354,4.695,4.841,2.341,4.146,4.500,6.122],
    'Gmean': [1.451,4.866,4.500,2.683,4.610,4.524,5.366],
    'BAC': [1.561,4.841,4.573,2.768,4.744,4.354,5.159],
})

kNN15s = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [3.817,5.280,4.720,4.829,4.463,2.695,2.195],
    'Precision': [5.524,5.000,4.390,4.915,4.232,2.244,1.695],
    'Recall': [1.366,4.549,4.841,2.671,3.610,4.988,5.976],
    'Gmean': [1.451,4.744,4.841,3.268,4.171,4.744,4.780],
    'BAC': [1.561,5.000,4.805,3.268,4.293,4.537,4.537],
})

kNN30s = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [3.963,5.549,4.793,4.707,4.427,2.451,2.110],
    'Precision': [5.598,5.159,4.378,4.866,4.146,2.146,1.707],
    'Recall': [1.341,4.488,4.866,2.671,3.488,5.134,6.012],
    'Gmean': [1.451,5.037,5.037,3.341,3.939,4.671,4.524],
    'BAC': [1.561,5.232,5.207,3.402,3.988,4.329,4.280],
})

kNN50s = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [4.037,5.671,4.866,4.488,4.280,2.573,2.085],
    'Precision': [5.646,5.134,4.439,4.756,4.268,2.098,1.659],
    'Recall': [1.341,4.512,4.768,2.854,3.256,5.183,6.085],
    'Gmean': [1.476,5.159,5.134,3.537,3.744,4.549,4.402],
    'BAC': [1.585,5.402,5.232,3.671,3.963,4.183,3.963],
})

kNN5 = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [3.585,4.305,3.476,4.549,4.390,3.744,3.951],
    'Precision': [5.317,3.963,3.049,4.976,3.659,3.878,3.159],
    'Recall': [1.427,5.232,5.366,2.463,4.939,3.305,5.268],
    'Gmean': [1.537,5.061,4.866,2.720,5.110,3.427,5.280],
    'BAC': [1.659,5.012,4.841,2.780,5.024,3.415,5.268],
})

kNN15 = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [3.500,4.720,4.207,4.683,4.122,3.354,3.415],
    'Precision': [5.451,4.171,3.585,4.671,3.695,3.293,3.134],
    'Recall': [1.439,5.183,5.402,2.817,4.244,4.049,4.866],
    'Gmean': [1.549,4.939,5.037,3.268,4.439,3.915,4.854],
    'BAC': [1.610,5.061,4.902,3.244,4.329,4.134,4.720],
})

kNN30 = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [3.622,4.988,4.159,4.585,4.159,3.280,3.207],
    'Precision': [5.549,4.378,3.524,4.585,3.744,3.037,3.183],
    'Recall': [1.415,5.098,5.451,2.915,3.951,4.524,4.646],
    'Gmean': [1.524,5.159,5.061,3.415,4.134,4.207,4.500],
    'BAC': [1.585,5.256,5.232,3.402,4.037,4.134,4.354],
})

kNN50 = pd.DataFrame({
    'group': ['kNN', 'OSB', 'OKNORAU', 'DESE-C', 'ODESE-C', 'DESE-A', 'ODESE-A'],
    'F1 score': [3.768,5.085,4.207,4.244,4.061,3.183,3.451],
    'Precision': [5.549,4.378,3.622,4.402,3.915,2.927,3.207],
    'Recall': [1.415,5.134,5.366,3.061,3.695,4.805,4.524],
    'Gmean': [1.573,5.183,5.256,3.512,3.866,4.500,4.110],
    'BAC': [1.634,5.329,5.207,3.524,3.890,4.354,4.061],
})

CART5s = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [3.415,3.768,3.915,5.622,5.768,2.524,2.988],
    'Precision': [3.683,4.659,4.878,5.793,5.256,1.793,1.939],
    'Recall': [3.146,2.488,2.561,3.793,4.110,5.817,6.085],
    'Gmean': [3.049,2.598,2.744,4.280,4.817,5.183,5.329],
    'BAC': [3.073,2.537,2.683,4.744,5.110,4.695,5.159],
})

CART15s = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [3.793,4.134,4.585,5.366,5.610,1.927,2.585],
    'Precision': [3.939,5.463,5.695,4.805,4.805,1.512,1.780],
    'Recall': [2.756,2.488,2.854,3.683,4.195,5.805,6.220],
    'Gmean': [2.695,2.646,3.146,4.585,4.951,4.634,5.341],
    'BAC': [2.720,2.829,3.329,4.707,5.098,4.268,5.049],
})

CART30s = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [3.951,4.232,4.793,5.244,5.537,1.927,2.317],
    'Precision': [3.976,5.805,5.659,4.512,4.659,1.634,1.756],
    'Recall': [2.573,2.317,3.134,3.659,4.171,5.854,6.293],
    'Gmean': [2.659,2.720,3.476,4.659,5.195,4.390,4.902],
    'BAC': [2.854,2.854,3.732,4.695,5.256,3.951,4.659],
})

CART50s = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'DESIRE-C', 'ODESIRE-C', 'DESIRE-A', 'ODESIRE-A'],
    'F1 score': [4.073,4.256,4.817,5.244,5.537,1.951,2.122],
    'Precision': [4.098,5.793,5.695,4.585,4.463,1.634,1.732],
    'Recall': [2.732,2.280,3.061,3.610,4.146,5.829,6.341],
    'Gmean': [2.951,2.793,3.476,4.756,5.244,4.146,4.634],
    'BAC': [3.000,2.915,3.598,4.976,5.366,3.805,4.341],
})


# ------- PART 1: Create background

# number of variable
df = kNN50s
categories = list(df)[1:]
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)

for label in ax.get_xticklabels():
    label.set_rotation(120)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks(
        # [0.45,0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9],
        # ["45%","50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%"],
        [0,1, 2, 3, 4, 5, 6, 7],
        ["0", "1", "2", "3", "4", "5", "6", "7"],
        fontsize=6,
    )
plt.ylim(0.0, 7.0)



# ------- PART 2: Add plots

# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable

# Ind1
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color="#704613")
ax.plot(angles, values, linewidth=1, linestyle='solid', label="kNN", color="#704613")
# ax.fill(angles, values, 'b', alpha=0.1)

# Ind2
values = df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='#248c5f')
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OSB", color='#248c5f')
# ax.fill(angles, values, 'r', alpha=0.1)

# Ind2
values = df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='#1765a3')
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OKNORAU", color='#1765a3')

values = df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='#e3a844')
ax.plot(angles, values, linewidth=1, linestyle='dashed', label="DESIRE-C", color='#e3a844')

values = df.loc[4].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='#efa61a')
ax.plot(angles, values, linewidth=1, linestyle='solid', label="ODESIRE-C", color='#efa61a')

values = df.loc[5].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='red')
ax.plot(angles, values, linewidth=1, linestyle='dashed', label="DESIRE-A", color='#e34844')

values = df.loc[6].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='red')
ax.plot(angles, values, linewidth=1, linestyle='solid', label="ODESIRE-A", color='#e34844')


# Add legend
plt.legend(loc="lower center", ncol=4, columnspacing=1, frameon=False, bbox_to_anchor=(0.5, -0.2))
# Add a title
plt.title("Mean ranks for kNN, pool size = 50", size=11, y=1.08)

plt.savefig("plots2/kNN50s.eps", bbox_inches='tight')
