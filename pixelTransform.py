import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt

root = r"D:\SpatioTemporalChanges\dataset"

palette = {
    0: (0, 0, 0),        # black -> No information
    1: (255, 255, 0),    # yellow -> Artificial surfaces
    2: (0, 255, 0),      # green -> Agricultural areas
    3: (255, 0, 0),      # red -> Forest
    4: (255, 0, 255),    # purple -> Wetlands
    5: (0, 0, 255),      # blue -> Water
}


def label_to_rgb(label):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, v in palette.items(): 
        rgb[label == k] = v
    return rgb

def process_region(year_folder, visual_folder):
    for region in os.listdir(year_folder):
        src_path = os.path.join(year_folder, region)
        dst_path = os.path.join(visual_folder, region)

        os.makedirs(dst_path, exist_ok=True)

        print(f"Processing {region}...")

        for file in os.listdir(src_path):
            full_src = os.path.join(src_path, file)
            full_dst = os.path.join(dst_path, file.replace(".tif", ".png"))

            with rasterio.open(full_src) as src:
                img = src.read(1)

            rgb = label_to_rgb(img)
            plt.imsave(full_dst, rgb)

process_region(
    os.path.join(root, "labels_land_cover_2006", "2006"),
    os.path.join(root, "visual_land_cover_2006")
)

process_region(
    os.path.join(root, "labels_land_cover_2012", "2012"),
    os.path.join(root, "visual_land_cover_2012")
)