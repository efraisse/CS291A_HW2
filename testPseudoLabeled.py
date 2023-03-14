import pandas as pd
import os
import pathlib 
import matplotlib
from PIL import Image
import random

file_path = pathlib.Path(__file__).parent / "data" / "ti_500K_pseudo_labeled"

obj = pd.read_pickle(file_path)

print(len(obj["data"]))
print(len(obj["extrapolated_targets"]))

index = random.randint(0, 500000)
print(f"index: {index}")
# for i in range(0, 1):
img = Image.fromarray(obj["data"][index], 'RGB')
img.save('firstpicturedataset.png')
print(obj["extrapolated_targets"][index])
img.show()
    
print(obj.keys())
#print(obj)