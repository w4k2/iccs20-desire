# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

# Set data
GNB50 = pd.DataFrame({
    'group': ['F1 score', 'Precision', 'Recall'],
    'GNB': [2.049,1.902,4.073],
    'OSB': [2.232,1.756,5.378],
    'OKNORAU': [3.793,3.585,4.634],
    'TBAC': [5.939,6.476,2.707],
    'OTBAC': [5.549,5.841,3.134],
    'TBAW': [4.171,4.463,3.866],
    'OTBAW': [4.268,3.976,4.207],
})

GNB50 = pd.DataFrame({
    'group': ['GNB', 'OSB', 'OKNORAU', 'TBAC', 'OTBAC', 'TBAW', 'OTBAW'],
    'F1 score': [2.049,2.232,3.793,5.939,5.549,4.171,4.268],
    'Precision': [1.902,1.756,3.585,6.476,5.841,4.463,3.976],
    'Recall': [4.073,5.378,4.634,2.707,3.134,3.866,4.207],
    'Gmean': [2.341,2.646,4.841,4.671,4.939,4.256,4.305],
    'BAC': [2.195,2.695,4.573,4.805,5.000,4.256,4.476],
})


# ------- PART 1: Create background

# number of variable
df = GNB50
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
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.15)
ax.plot(angles, values, linewidth=1, linestyle='solid', label="GNB", color='plum')
# ax.fill(angles, values, 'b', alpha=0.1)

# Ind2
values = df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.15)
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OSB", color='green')
# ax.fill(angles, values, 'r', alpha=0.1)

# Ind2
values = df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.15)
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OKNORAU", color='blue')

values = df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.15)
ax.plot(angles, values, linewidth=1, linestyle='dashed', label="TBAC", color='orange')

values = df.loc[4].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.15)
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OTBAC", color='orange')

values = df.loc[5].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.15, color='red')
ax.plot(angles, values, linewidth=1, linestyle='dashed', label="TBAW", color='red')

values = df.loc[6].drop('group').values.flatten().tolist()
values += values[:1]
# ax.fill(angles, values, linewidth=1, linestyle='solid', alpha=0.15, color='red')
ax.plot(angles, values, linewidth=1, linestyle='solid', label="OTBAW", color='red')


# Add legend
plt.legend(loc="lower center", ncol=4, columnspacing=1, frameon=False, bbox_to_anchor=(0.5, -0.2))
# Add a title
plt.title("Mean ranks for GNB, pool size = 50", size=11, y=1.08)


plt.savefig("GNB50.eps", bbox_inches='tight')
