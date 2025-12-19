import argparse, os, random, shutil, glob, pathlib
def yolo_txt(img_path, labels_root):
    stem = pathlib.Path(img_path).stem
    cand = os.path.join(labels_root, stem + ".txt")
    return cand if os.path.isfile(cand) else None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="datasets/doggo/images_all")
    ap.add_argument("--labels", default="datasets/doggo/labels_all")
    ap.add_argument("--out", default="datasets/doggo")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6
    for split in ("train","val","test"):
        os.makedirs(os.path.join(args.out, f"images/{split}"), exist_ok=True)
        os.makedirs(os.path.join(args.out, f"labels/{split}"), exist_ok=True)
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(args.images, ext)))
    imgs = sorted(imgs)
    random.Random(args.seed).shuffle(imgs)
    n = len(imgs)
    n_train = int(n*args.train)
    n_val   = int(n*args.val)
    splits = {"train": imgs[:n_train], "val": imgs[n_train:n_train+n_val], "test": imgs[n_train+n_val:]}
    moved = {k:0 for k in splits}
    for split, paths in splits.items():
        for img in paths:
            lbl = yolo_txt(img, args.labels)
            if lbl is None:
                print(f"[WARN] Sem label para {img}; pulando.")
                continue
            shutil.copy2(img, os.path.join(args.out, f"images/{split}", os.path.basename(img)))
            shutil.copy2(lbl, os.path.join(args.out, f"labels/{split}", os.path.basename(lbl)))
            moved[split]+=1
    print("Feito:", moved)
if __name__ == "__main__":
    main()
