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
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["models/keepfit-synfundus+mm.pth", "models/keepfit-ffa-ir+mm.pth"])
    parser.add_argument("--weights", type=float, nargs="+", 
                        default=[2, 1])
    parser.add_argument("--adv_label_idx", type=int, default=-1)
    #parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()
                        
args = parse_args()

models = []
num_models = len(args.models)
if len(args.models) != len(args.weights):
    print(f"Number of models {len(args.weights)} != number of weights {len(args.models)}!")
    exit(0)
model_weights = np.expand_dims(np.array(args.weights, dtype=float), 1)
model_weights /= model_weights.sum()

for model_path in args.models:
    model = KeepFITModel(from_checkpoint=True, weights_path=model_path)
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

all_input_filepaths = []
for input in args.input:
    if os.path.isfile(input):
        input_filepaths = [input]
    else:
        input_filepaths = os.listdir(input)
        input_filepaths = [ os.path.join(input, input_filepath) for input_filepath in input_filepaths]
        input_filepaths = sorted(input_filepaths)

    all_input_filepaths.extend(input_filepaths)
    
for input_filepath in all_input_filepaths:
    image = image_np = np.array(Image.open(input_filepath))
    input_filestem = os.path.splitext(input_filepath)[0]

    print(f"{input_filepath}:")

    if args.adv_label_idx >= 0:
        # Resize image to 512*512. Since synfundus images are already 512*512,
        # there's no scaling in this case, and image_grad is already spatially aligned with image.
        image = model.preprocess_image(image)
        image.requires_grad = True
        do_bp = True
        do_preprocess_image = False
    else:
        do_bp = False
        do_preprocess_image = True

    all_scores = []
    all_probs  = []

    for mi, model in enumerate(models):
        with torch.set_grad_enabled(do_bp):
            probs, logits = model(image, labels, do_preprocess_image=do_preprocess_image)
            
        probs_np    = probs[0].detach().cpu().numpy()
        logits_np   = logits[0].detach().cpu().numpy()

        print(" ", to_str(np.arange(len(labels)), fmt=" 5d"))

        print(" ", f"Scores {mi}:")
        print(" ", to_str(logits_np)) # [[-0.32  -2.782  3.164  4.388  5.919  6.639  6.579 10.478]]
        print(" ", f"Probs {mi}:")
        print(" ", to_str(probs_np))  # [[0.      0.     0.001  0.002  0.01   0.02   0.019  0.948]]
        print(" ", f"Class {mi}:")
        print(" ", np.argmax(probs_np), labels[np.argmax(probs_np)])
        print()    

        all_scores.append(logits_np)
        all_probs.append(probs_np)

        if do_bp:
            contrib_weighted_by_pixel_values = False # True
            ignore_neg_contrib = False

            loss = logits[:, args.adv_label_idx]
            loss.backward()

    if len(all_scores) > 1:
        all_scores = np.array(all_scores)
        all_probs  = np.array(all_probs)
        avg_scores = (all_scores * model_weights).sum(axis=0)
        avg_probs  = (all_probs  * model_weights).sum(axis=0)

        print(to_str(np.arange(len(labels)), fmt=" 5d"))
        print(f"Avg Scores:")
        print(to_str(avg_scores)) # [[-0.32  -2.782  3.164  4.388  5.919  6.639  6.579 10.478]]
        print(f"Avg Probs:")
        print(to_str(avg_probs))  # [[0.      0.     0.001  0.002  0.01   0.02   0.019  0.948]]
        print(f"Avg Class:")
        print(np.argmax(avg_probs), labels[np.argmax(avg_probs)])
        print()    

        # image_grad: [3, 512, 512]
        # The passed gradients are towards decreasing the objective function output
        # which are to be subtracted from the feature values
        # Hence the larger (positive) numbers image_grad are, the more changes
        # the corresponding features can cause
        # Hence more contributions from a unit-valued feature in slice_feat
        image_grad = image.grad
        if contrib_weighted_by_pixel_values:
            image_grad = image_grad * image
        image_grad = image_grad[0].detach().cpu().numpy()

        if ignore_neg_contrib:
            # Ignore negative gradients
            image_grad[image_grad < 0] = 0
        else:
            image_grad = np.abs(image_grad)
        # image_grad: [1, 512, 512]
        image_grad = image_grad.sum(axis=0, keepdims=True)

        image_grad = (image_grad - image_grad.min()) / (image_grad.max() - image_grad.min() + 1e-6)
        # image_grad: [512, 512, 1]
        image_grad = np.uint8(255 * image_grad).transpose(1, 2, 0)
        # Overlay the gradient on the image
        image_grad_map  = cv2.applyColorMap(image_grad, cv2.COLORMAP_JET)
        image_with_grad = cv2.addWeighted(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), 0.8, image_grad_map, 0.5, 0)
        save_filepath = f"{input_filestem}_grad-{num_models}m.png"
        if os.path.exists(save_filepath):
            counter = 1
            while True:
                save_filepath = f"{input_filestem}_grad-{num_models}m{counter}.png"
                if not os.path.exists(save_filepath):
                    break
                counter += 1
            
        cv2.imwrite(save_filepath, image_with_grad)
        print(f"Gradient map saved to {save_filepath}")
