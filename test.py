from PIL import Image
import numpy as np

# Import FLAIR
from KeepFIT.KeepFIT_CFP.keepfit import KeepFITModel
import argparse
import os
import torch
import cv2

def to_str(array, fmt=" 3.2f"):
    formatted_elements = [("{:" + fmt + "}").format(val) for val in np.nditer(array)]
    return ' '.join(formatted_elements)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--weight", type=str, nargs="+",
                        default=["weights/keepfit-ffa-ir+mm.pth"])
    #parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()
                        
args = parse_args()

models = []
for weight in args.weight:
    model = KeepFITModel(from_checkpoint=True, weights_path=weight)
    model.eval()
    models.append(model)

# Load image and set target categories 
# (if the repo is not cloned, download the image and change the path!)

labels = [
    "Normal Healthy",                       # 0
    "Diabetic Retinopathy",                 # 1
    "Age-related Macular Degeneration",     # 2
    "Anomalies of the Optic Nerve",         # 3
    "Choroidal Retinal Vascular",           # 4
    "Diabetic Macular Edema",               # 5
    "Epimacular Membrane",                  # 6
    "Glaucoma",                             # 7
    "Hypertensive Retinopathy",             # 8
    "Myopia",                               # 9
    "Retinal Vein Occlusion",               # 10
    "Macular lesion",                       # 11
    "Hemorrhagic spots",                    # 12
    "Tessellated"                           # 13
]

print()
print("Labels:")
for i, label in enumerate(labels):
    print(f" {i}: {label}")
print()

all_input_files = []
for input in args.input:
    if os.path.isfile(input):
        input_files = [input]
    else:
        input_files = os.listdir(input)
        input_files = [ os.path.join(input, input_file) for input_file in input_files]
        input_files = sorted(input_files)

    all_input_files.extend(input_files)
    
for input_file in all_input_files:
    image = np.array(Image.open(input_file))
    all_probs = []
    all_logits = []

    print(f"{input_file} model:")

    for i, model in enumerate(models):
        with torch.no_grad():
            probs, logits = model(image, labels)
        all_probs.append(probs)
        all_logits.append(logits)

        print(" ", to_str(np.arange(len(labels)), fmt=" 5d"))

        print(f"  scores {i}:")
        print(" ", to_str(logits)) # [[-0.32  -2.782  3.164  4.388  5.919  6.639  6.579 10.478]]
        print(f"  probs {i}:")
        print(" ", to_str(probs))  # [[0.      0.     0.001  0.002  0.01   0.02   0.019  0.948]]
        print(f"  Class {i}:")
        print(" ", np.argmax(probs), labels[np.argmax(probs)])
        print()    

    if len(models) > 1:
        # Average the logits
        avg_logits = np.mean(all_logits, axis=0)
        avg_probs =  np.mean(all_probs, axis=0)
        print(f" avg scores:")
        print("", to_str(avg_logits))
        print(f" avg probs:")
        print("", to_str(avg_probs))
        print(f" Class:")
        print(np.argmax(avg_probs), labels[np.argmax(avg_probs)])
        print()
