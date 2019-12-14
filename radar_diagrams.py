# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

# Set data

GNB5 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.146,2.085,3.500,5.549,5.963,4.159,4.598],
    'Precision': [1.829,1.756,3.220,6.256,5.866,4.720,4.354],
    'Recall': [4.207,5.159,4.902,2.134,3.744,3.329,4.524],
    'Gmean': [2.341,2.695,4.183,4.695,5.890,3.622,4.573],
    'BAC': [2.317,2.634,3.963,4.720,5.976,3.671,4.720],
})

GNB15 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.049,2.037,3.646,5.915,5.817,4.146,4.390],
    'Precision': [1.902,1.683,3.366,6.573,5.866,4.512,4.098],
    'Recall': [4.073,5.012,4.439,2.585,3.549,3.683,4.659],
    'Gmean': [2.268,2.500,4.378,4.768,5.524,3.878,4.683],
    'BAC': [2.195,2.476,4.183,4.890,5.573,3.951,4.732],
})

GNB30 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.000,2.171,3.878,5.890,5.695,3.976,4.390],
    'Precision': [1.927,1.707,3.415,6.512,5.878,4.451,4.110],
    'Recall': [4.037,5.378,4.780,2.622,3.220,3.598,4.366],
    'Gmean': [2.220,2.659,4.854,4.598,5.012,4.000,4.659],
    'BAC': [2.171,2.634,4.659,4.817,5.110,3.976,4.634],
})

GNB50 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.049,2.232,3.793,5.939,5.549,4.171,4.268],
    'Precision': [1.902,1.756,3.585,6.476,5.841,4.463,3.976],
    'Recall': [4.073,5.378,4.634,2.707,3.134,3.866,4.207],
    'Gmean': [2.341,2.646,4.841,4.671,4.939,4.256,4.305],
    'BAC': [2.195,2.695,4.573,4.805,5.000,4.256,4.476],
})

CART5 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.561,2.927,3.024,5.268,5.512,4.232,4.476],
    'Precision': [2.463,4.195,4.293,5.256,5.256,3.073,3.463],
    'Recall': [3.122,2.744,2.768,3.915,4.415,5.317,5.720],
    'Gmean': [3.024,2.756,2.829,4.024,4.610,5.305,5.451],
    'BAC': [2.902,2.805,2.878,4.280,4.707,5.073,5.354],
})

CART15 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.805,3.195,3.659,5.268,5.463,3.512,4.098],
    'Precision': [2.927,5.256,5.500,4.671,4.671,2.573,2.402],
    'Recall': [2.927,2.683,2.927,3.756,4.366,5.146,6.195],
    'Gmean': [2.732,2.707,3.122,4.073,4.610,4.780,5.976],
    'BAC': [2.683,2.780,3.122,4.220,4.732,4.634,5.829],
})

CART30 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.951,3.427,4.012,5.171,5.390,3.439,3.610],
    'Precision': [3.171,5.805,5.439,4.500,4.378,2.329,2.378],
    'Recall': [2.841,2.305,3.268,3.695,4.354,5.366,6.171],
    'Gmean': [2.585,2.476,3.305,4.122,4.634,4.976,5.902],
    'BAC': [2.488,2.622,3.354,4.171,4.732,4.854,5.780],
})

CART50 = pd.DataFrame({
    'group': ['CART', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.951,3.512,4.293,5.171,5.171,3.390,3.512],
    'Precision': [3.366,5.817,5.866,4.329,4.134,2.232,2.256],
    'Recall': [2.805,2.378,3.244,3.659,4.207,5.500,6.207],
    'Gmean': [2.610,2.512,3.390,4.098,4.561,4.951,5.878],
    'BAC': [2.537,2.707,3.488,4.146,4.659,4.732,5.732],
})


# ------- PART 1: Create background

# number of variable
df = CART50
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
ax.plot(angles, values, linewidth=1, linestyle='solid', label="CART", color="#704613")
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
ax.plot(angles, values, linewidth=1, linestyle='dashed', label="TBAC", color='#e3a844')

values = df.loc[4].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='#efa61a')
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OTBAC", color='#efa61a')

values = df.loc[5].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='red')
ax.plot(angles, values, linewidth=1, linestyle='dashed', label="TBAW", color='#e34844')

values = df.loc[6].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.03, color='red')
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OTBAW", color='#e34844')


# Add legend
plt.legend(loc="lower center", ncol=4, columnspacing=1, frameon=False, bbox_to_anchor=(0.5, -0.2))
# Add a title
plt.title("Mean ranks for CART, pool size = 50", size=11, y=1.08)


plt.savefig("CART50.eps", bbox_inches='tight')
