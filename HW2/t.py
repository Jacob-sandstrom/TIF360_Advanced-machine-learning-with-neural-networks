import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


data = pkl.load(open('simple_particle_dataset.pkl', 'rb'))

print(data.keys())

print(data["images"][0])


print(data["labels"][:17])

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(data["images"][i], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()