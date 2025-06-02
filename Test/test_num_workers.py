import time
from torch.utils.data import DataLoader
from dataset.faster_rcnn_dataset import FasterRCNNDataset
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def test_num_workers(max_workers=16, batch_size=8):
    train_img_path = os.path.join(PROJECT_ROOT, "dataset", "train", "images")
    train_lbl_path = os.path.join(PROJECT_ROOT, "dataset", "train", "labels")
    dataset1 = FasterRCNNDataset(train_img_path, train_lbl_path)

    results = []

    for nw in range(0, max_workers + 1):
        loader = DataLoader(dataset1, batch_size=batch_size, num_workers=nw, collate_fn=lambda x: tuple(zip(*x)))

        start = time.time()
        for i, (imgs, targets) in enumerate(loader):
            if i == 10:  # sadece 10 batch test edelim
                break
        duration = time.time() - start
        results.append((nw, duration))
        print(f"num_workers={nw} için süre: {duration:.4f} saniye")

    print("\n🏁 En hızlı num_workers değeri:")
    results.sort(key=lambda x: x[1])
    for nw, dur in results:
        print(f" - {nw} işçi: {dur:.4f} saniye")

test_num_workers()
