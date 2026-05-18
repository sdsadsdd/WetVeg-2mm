# WetVeg-2mm

WetVeg-2mm is an ultra-high-resolution UAV orthomosaic dataset for fine-grained riparian vegetation semantic segmentation. The dataset was constructed from UAV surveys over a representative riparian section of the Qing River in Guangxi, China, with a ground sampling distance of **2 mm**.

The dataset contains **2054** annotated image chips of size **1024 × 1024** pixels and provides pixel-level semantic labels for **17 semantic classes** in total, including **14 representative wetland plant classes**, together with **water**, **bareland**, and **background**.

This repository provides the dataset files, official split files, and original polygon annotations for reproducible research in fine-grained riparian vegetation segmentation.

> *This repository provides the benchmark code, official split files, metadata, and usage instructions. The full dataset is publicly available at Zenodo: [10.5281/zenodo.19849119].*

---

## Highlights

- Ultra-high-resolution UAV orthomosaic dataset at **2 mm** ground sampling distance  
- **2054** annotated image chips for fine-grained semantic segmentation  
- **17 semantic classes** in total  
- Official **train/val/test** split files for reproducible experiments  
- Original **Labelme JSON annotations** and semantic masks are both provided  

---

## 1. Dataset Overview

Riparian vegetation mapping is important for ecological monitoring, invasive species management, and restoration assessment. However, riparian plant communities often show fragmented patches, mixed boundaries, intra-class variation, and strong visual similarity among classes, making fine-grained semantic segmentation highly challenging. WetVeg-2mm was developed to provide a public benchmark for this task using ultra-high-resolution UAV orthomosaics.

### Basic information

- **Dataset name:** WetVeg-2mm  
- **Study area:** Representative riparian section of the Qing River, Guangxi, China  
- **Ground sampling distance:** 2 mm  
- **Image size:** 1024 × 1024 pixels  
- **Number of image chips:** 2054  
- **Annotation type:** Pixel-level semantic segmentation  
- **Number of semantic classes:** 17  

---

## 2. Dataset Structure

The dataset is organized as follows:

~~~text
WetVeg-2mm/
├── JPEGImages/
├── SegmentationClass/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── json/
├── Class_list.xlsx
├── README.md
└── LICENSE
~~~

### Folder description

- **JPEGImages/**  
  Stores all original image chips in JPEG format.

- **SegmentationClass/**  
  Stores the publicly released semantic segmentation masks in PNG format.

- **ImageSets/**  
  Stores the official split files:
  - `train.txt`
  - `val.txt`
  - `test.txt`  
  Each file records the image names belonging to the corresponding subset.

- **json/**  
  Stores the publicly released Labelme polygon annotations.

- **Class_list.xlsx**  
  Stores the class names and corresponding label IDs.


---

## 3. File Correspondence

All image files, mask files, and JSON annotation files share the same filename stem, with different file extensions only. Therefore, the correspondence among the original image, semantic mask, and polygon annotation can be established directly by filename.

For example:

- `JPEGImages/xxxx.jpg`
- `SegmentationClass/xxxx.png`
- `json/xxxx.json`

all correspond to the same sample.

---

## 4. Official Data Split

The dataset provides an official split for benchmark experiments:

- **Training set:** 1227 samples  
- **Validation set:** 405 samples  
- **Test set:** 422 samples  

The corresponding filenames are listed in:

- `ImageSets/train.txt`
- `ImageSets/val.txt`
- `ImageSets/test.txt`

---

## 5. Semantic Classes

WetVeg-2mm contains **17 semantic classes** in total, including **14 representative wetland plant classes**, together with **water**, **bareland**, and **background**.

| Label ID | Class Name | Scientific Name |
|---------:|------------|-----------------|
| 0 | Background | Background |
| 1 | Water | Water |
| 2 | Bareland | Bareland |
| 3 | Bambusa | *Bambusa multiplex* |
| 4 | Rhodomyrtus | *Rhodomyrtus tomentosa* |
| 5 | Aeschynomene | *Aeschynomene indica* |
| 6 | Pongamia | *Pongamia pinnata* |
| 7 | Bidens | *Bidens alba* |
| 8 | Syzygium | *Syzygium nervosum* |
| 9 | Eichhornia | *Eichhornia crassipes* |
| 10 | Colocasia | *Colocasia esculentum var. antiquorum* |
| 11 | Alternanthera | *Alternanthera philoxeroides* |
| 12 | Musa | *Musa balbisiana* |
| 13 | Diplazium | *Diplazium esculentum* |
| 14 | Arundo | *Arundo donax* |
| 15 | Pueraria | *Pueraria montana var. lobata* |
| 16 | Phragmites | *Phragmites australis* |

---

## 6. Data Format

### 6.1 Image chips
All original image chips are stored in `JPEGImages/` in **JPEG** format.

### 6.2 Semantic masks
All semantic masks are stored in `SegmentationClass/` in **PNG** format. Each pixel value corresponds to a semantic class ID.

### 6.3 Polygon annotations
All original manual annotations are stored in `json/` in **Labelme JSON** format.

---

## 7. Recommended Usage

For semantic segmentation experiments, the recommended data usage is:

- Use images in `JPEGImages/` as model inputs
- Use masks in `SegmentationClass/` as ground truth labels
- Use `train.txt`, `val.txt`, and `test.txt` for official benchmark splitting

The JSON files in `json/` are provided as the original annotation source and can be used for further label checking, re-exporting masks, or annotation extension.

---

## 8. Benchmark Setting

WetVeg-2mm was benchmarked using four representative semantic segmentation models:

- U-Net
- Attention U-Net
- DeepLabV3+
- PSPNet

The evaluation metrics include:

- per-class IoU
- mIoU
- mDice
- Pixel Accuracy (PA)
- Precision
- Recall

---

## 9. Data Access

The publicly released version of WetVeg-2mm is available at Zenodo:

**DOI:** 10.5281/zenodo.19849119

The public release includes:
- image chips
- training and validation masks
- training and validation JSON annotations
- class mapping file
- official split files

To support controlled benchmark evaluation, the **test annotations are not publicly released**. This includes:
- test masks in `SegmentationClass`
- test JSON annotation files in `json`

Researchers who would like to evaluate their methods on the test set may contact the authors for controlled testing.
---

## 10. Citation

If you use WetVeg-2mm in your research, please cite the associated paper.
---

## 11. License

This dataset is released under the **CC BY 4.0** license.

---

## 12. Contact

For questions regarding the dataset, please contact:

- **Name:** Huapeng Qin
- **Affiliation:** Eco-environment and Resource Efficiency Research Laboratory, School of Environment and Energy, Peking University Shenzhen Graduate School, Shenzhen 518055, China
- **Email:** qinhp@pkusz.edu.cn

---

## 13. Notes

- All filenames in `JPEGImages/`, `SegmentationClass/`, and `json/` correspond one-to-one by filename stem.
- Official train/val/test split files are provided in `ImageSets/`.
- The publicly released version does **not** include test annotation masks or test JSON labels.
- Users should note that the dataset exhibits clear class imbalance and patch-scale variation across categories.
