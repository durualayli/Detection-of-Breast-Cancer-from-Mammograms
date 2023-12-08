import pandas as pd
import numpy as np
import glob
import os
import pydicom
from PIL import Image
import shutil

data = pd.read_csv("mass_case_description_train_set.csv")

grouped_imgview = data.groupby(['image view'])
MLO_images = grouped_imgview.get_group("MLO")
CC_images = grouped_imgview.get_group("CC")

grouped_pathology = data.groupby(['pathology'])
malignant_images = grouped_pathology.get_group("MALIGNANT")
benign_wout_images = grouped_pathology.get_group("BENIGN_WITHOUT_CALLBACK")
benign_images = grouped_pathology.get_group("BENIGN")

grouped_MLO_pathology = MLO_images.groupby(['pathology'])
MLO_mal = grouped_MLO_pathology.get_group("MALIGNANT")
MLO_ben_wout = grouped_MLO_pathology.get_group("BENIGN_WITHOUT_CALLBACK")
MLO_ben = grouped_MLO_pathology.get_group("BENIGN")

grouped_CC_pathology = CC_images.groupby(['pathology'])
CC_mal = grouped_CC_pathology.get_group("MALIGNANT")
CC_ben_wout = grouped_CC_pathology.get_group("BENIGN_WITHOUT_CALLBACK")
CC_ben = grouped_CC_pathology.get_group("BENIGN")

MLO_mal_full_img = MLO_mal.loc[:,"image file path"]
MLO_ben_wout_full_img = MLO_ben_wout.loc[:,"image file path"]
MLO_ben_full_img = MLO_ben.loc[:,"image file path"]
CC_mal_full_img = CC_mal.loc[:,"image file path"]
CC_ben_wout_full_img = CC_ben_wout.loc[:,"image file path"]
CC_ben_full_img = CC_ben.loc[:,"image file path"]

MLO_mal_full_pat = [os.path.dirname(os.path.dirname(os.path.dirname(r))) for r in MLO_mal_full_img]
MLO_ben_full_pat = [os.path.dirname(os.path.dirname(os.path.dirname(r))) for r in MLO_ben_full_img]
MLO_ben_wout_full_pat = [os.path.dirname(os.path.dirname(os.path.dirname(r))) for r in MLO_ben_wout_full_img]
CC_mal_full_pat = [os.path.dirname(os.path.dirname(os.path.dirname(r))) for r in CC_mal_full_img]
CC_ben_full_pat = [os.path.dirname(os.path.dirname(os.path.dirname(r))) for r in CC_ben_full_img]
CC_ben_wout_full_pat = [os.path.dirname(os.path.dirname(os.path.dirname(r))) for r in CC_ben_wout_full_img]

MLO_mal_full_p = [os.path.join(r"/Users/ahmetalayli/Desktop/DL/CBIS-DDSM/"+r) for r in MLO_mal_full_pat]
MLO_ben_wout_full_p = [os.path.join(r"/Users/ahmetalayli/Desktop/DL/CBIS-DDSM/"+r) for r in MLO_ben_wout_full_pat]
MLO_ben_full_p = [os.path.join(r"/Users/ahmetalayli/Desktop/DL/CBIS-DDSM/"+r) for r in MLO_ben_full_pat]
CC_mal_full_p = [os.path.join(r"/Users/ahmetalayli/Desktop/DL/CBIS-DDSM/"+r) for r in CC_mal_full_pat]
CC_ben_wout_full_p = [os.path.join(r"/Users/ahmetalayli/Desktop/DL/CBIS-DDSM/"+r) for r in CC_ben_wout_full_pat]
CC_ben_full_p = [os.path.join(r"/Users/ahmetalayli/Desktop/DL/CBIS-DDSM/"+r) for r in CC_ben_full_pat]

MLO_mal_full_path = []
MLO_ben_wout_full_path = []
MLO_ben_full_path = []
CC_mal_full_path = []
CC_ben_wout_full_path = []
CC_ben_full_path = []

for r in MLO_mal_full_p:
    pathway = r
    MLO_mal_full_path.append(glob.glob(pathway + "/**/*.dcm", recursive = True))

for r in MLO_ben_wout_full_p:
    pathway = r
    MLO_ben_wout_full_path.append(glob.glob(pathway + "/**/*.dcm", recursive = True))

for r in MLO_ben_full_p:
    pathway = r
    MLO_ben_full_path.append(glob.glob(pathway + "/**/*.dcm", recursive = True))

for r in CC_mal_full_p:
    pathway = r
    CC_mal_full_path.append(glob.glob(pathway + "/**/*.dcm", recursive = True))

for r in CC_ben_wout_full_p:
    pathway = r
    CC_ben_wout_full_path.append(glob.glob(pathway + "/**/*.dcm", recursive = True))

for r in CC_ben_full_p:
    pathway = r
    CC_ben_full_path.append(glob.glob(pathway + "/**/*.dcm", recursive = True))

MLO_mal_full_folder = "./MLO-mal-full-images"
MLO_ben_wout_full_folder = "./MLO-ben-wout-full-images"
MLO_ben_full_folder = "./MLO-ben-full-images"
CC_mal_full_folder = "./CC-mal-full-images"
CC_ben_wout_full_folder = "./CC-ben-wout-full-images"
CC_ben_full_folder = "./CC-ben-full-images"
try:
    os.mkdir(MLO_mal_full_folder)
    print("Folder %s created!" % MLO_mal_full_folder)
except FileExistsError:
    print("Folder %s already exists" % MLO_mal_full_folder)

try:
    os.mkdir(MLO_ben_wout_full_folder)
    print("Folder %s created!" % MLO_ben_wout_full_folder)
except FileExistsError:
    print("Folder %s already exists" % MLO_ben_wout_full_folder)

try:
    os.mkdir(MLO_ben_full_folder)
    print("Folder %s created!" % MLO_ben_full_folder)
except FileExistsError:
    print("Folder %s already exists" % MLO_ben_full_folder)

try:
    os.mkdir(CC_mal_full_folder)
    print("Folder %s created!" % CC_mal_full_folder)
except FileExistsError:
    print("Folder %s already exists" % CC_mal_full_folder)

try:
    os.mkdir(CC_ben_wout_full_folder)
    print("Folder %s created!" % CC_ben_wout_full_folder)
except FileExistsError:
    print("Folder %s already exists" % CC_ben_wout_full_folder)

try:
    os.mkdir(CC_ben_full_folder)
    print("Folder %s created!" % CC_ben_full_folder)
except FileExistsError:
    print("Folder %s already exists" % CC_ben_full_folder)

def grouping_pngs(path, folder):

    for r in path:
        if len(r) > 0:
            dicom_img = r[0]
            ds = pydicom.dcmread(dicom_img)
            new_image = ds.pixel_array.astype(float)
            scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
            scaled_image = np.uint8(scaled_image)
            final_image = Image.fromarray(scaled_image)
            file_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(dicom_img))))+".png"
        
        
        try:
            final_image.save(file_name)
            shutil.move(file_name,folder)
        except FileExistsError and OSError:
            print ("File exists")
            os.remove(file_name)
        
grouping_pngs(MLO_mal_full_path,MLO_mal_full_folder)
grouping_pngs(MLO_ben_wout_full_path,MLO_ben_wout_full_folder)
grouping_pngs(MLO_ben_full_path,MLO_ben_full_folder)
grouping_pngs(CC_mal_full_path,CC_mal_full_folder)
grouping_pngs(CC_ben_wout_full_path,CC_ben_wout_full_folder)
grouping_pngs(CC_ben_full_path,CC_ben_full_folder)