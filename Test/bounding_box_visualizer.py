import matplotlib.pyplot as plt

def visualize_sample(dataset, index):
    image_tensor, targets = dataset[index]

    # Tensor → NumPy
    image = image_tensor.permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
    image = (image * 255).astype(np.uint8)

    # Bounding box çiz
    for target in targets:
        cls, x1, y1, x2, y2 = target
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = (0, 255, 0) if cls == 0 else (255, 0, 0)
        label = "Hexagon" if cls == 0 else "Triangle"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Matplotlib ile göster
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
