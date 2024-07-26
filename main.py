import pandas as pd
import lpips
import cv2
import torch
import numpy as np
import os
from tqdm import tqdm


dataset = pd.read_csv("koniq10k_scores_and_distributions.csv")
max_mos = dataset["MOS"].values.max()
min_mos = dataset["MOS"].values.min()

esrgan_scores = {}
esrgan_scores["image_name"] = []
esrgan_scores["MOS"] = []

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

images = os.listdir("1024x768")
for image_name in tqdm(images):
    esrgan_scores["image_name"].append(image_name)
    img0 = cv2.imread(f"1024x768/{image_name}")
    img1 = cv2.imread(f"ESRGAN/results/{image_name[:-4]}_rlt.png")

    img0 = img0.reshape(1, 3, 768, 1024)
    img1 = img1.reshape(1, 3, 768, 1024)

    img0 = (img0 / 127.5) - 1
    img1 = (img1 / 127.5) - 1

    img0 = img0.astype(np.float32)
    img1 = img1.astype(np.float32)

    img0 = torch.tensor(img0)
    img1 = torch.tensor(img1)

    d = loss_fn_alex(img0, img1)
    esrgan_scores["MOS"].append(d * (max_mos - min_mos) + min_mos)
    df = pd.DataFrame(esrgan_scores)
    df.to_csv("ESRGAN_scores.csv")
