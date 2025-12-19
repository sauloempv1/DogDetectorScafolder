import os
import argparse
import cv2
import glob
from tqdm import tqdm
import albumentations as A


# ----------------------------
# Utilidades para leitura YOLO
# ----------------------------
def load_yolo_labels(label_path):
    boxes = []
    classes = []
    if not os.path.exists(label_path):
        return boxes, classes

    with open(label_path, "r") as f:
        for line in f.readlines():
            c, x, y, w, h = map(float, line.strip().split())
            classes.append(int(c))
            boxes.append([x, y, w, h])
    return boxes, classes


def save_yolo_labels(label_path, boxes, classes):
    with open(label_path, "w") as f:
        for cls, (x, y, w, h) in zip(classes, boxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# ----------------------------
# Pipeline de Data Augmentation
# ----------------------------
def get_augmentation():
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
                p=0.7
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.7),
            A.HueSaturationValue(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(5, 30), p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
    )


# ----------------------------
# Processo principal
# ----------------------------
def run(src_images, src_labels, out_images, out_labels, n, copy_original):

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    aug = get_augmentation()

    image_paths = sorted(glob.glob(os.path.join(src_images, "*.jpg")) +
                         glob.glob(os.path.join(src_images, "*.png")))

    for img_path in tqdm(image_paths, desc="Augmentando"):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(src_labels, img_name.rsplit(".", 1)[0] + ".txt")

        # Carrega imagem e labels
        image = cv2.imread(img_path)
        boxes, classes = load_yolo_labels(label_path)

        # Copia imagem original
        if copy_original:
            out_img_path = os.path.join(out_images, img_name)
            out_label_path = os.path.join(out_labels, img_name.rsplit(".", 1)[0] + ".txt")

            cv2.imwrite(out_img_path, image)
            save_yolo_labels(out_label_path, boxes, classes)

        # Gera N augmentações
        for i in range(n):
            augmented = aug(image=image, bboxes=boxes, class_labels=classes)

            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

            new_name = img_name.rsplit(".", 1)[0] + f"_aug{i}." + img_name.rsplit(".", 1)[1]

            out_img_path = os.path.join(out_images, new_name)
            out_label_path = os.path.join(out_labels, new_name.replace(".jpg", ".txt").replace(".png", ".txt"))

            cv2.imwrite(out_img_path, aug_img)
            save_yolo_labels(out_label_path, aug_boxes, aug_classes)


# ----------------------------
# Main + CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Data Augmentation with Albumentations")

    parser.add_argument("--src-images", required=True)
    parser.add_argument("--src-labels", required=True)
    parser.add_argument("--out-images", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("-n", type=int, default=3, help="Número de augmentações por imagem")
    parser.add_argument("--copy-original", action="store_true", help="Copiar imagem original")

    args = parser.parse_args()

    run(
        src_images=args.src_images,
        src_labels=args.src_labels,
        out_images=args.out_images,
        out_labels=args.out_labels,
        n=args.n,
        copy_original=args.copy_original
    )