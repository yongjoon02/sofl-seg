import matplotlib.pyplot as plt


def visualize_dataset(loader, dataset_name='octa', num_samples=5):
    # 첫 번째 샘플로 어떤 키가 있는지 확인
    first_item = next(iter(loader))
    has_label_prob = "label_prob" in first_item
    has_label_sauna = "label_sauna" in first_item

    # 컬럼 수 결정
    num_cols = 2  # image, label는 항상 있음
    if has_label_prob:
        num_cols += 1
    if has_label_sauna:
        num_cols += 1

    fig_width = 3 * num_cols
    fig_height = 4 * num_samples

    # num_samples가 1일 때와 그 이상일 때를 구분해서 처리
    if num_samples == 1:
        fig, axes = plt.subplots(1, num_cols, figsize=(fig_width, fig_height))
        axes = axes.reshape(1, -1)  # 2차원 배열로 변환
    else:
        fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, fig_height))

    for i, item in enumerate(loader):
        if i == num_samples:
            break

        image = item["image"][0].squeeze(0)
        label = item["label"][0].squeeze(0)

        # 값 범위 체크 및 출력
        print(f"Sample {i+1} - {item['name'][0]}:")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Label range: [{label.min():.3f}, {label.max():.3f}]")

        if has_label_prob:
            label_prob = item["label_prob"][0].squeeze(0)
            print(f"  Label_prob range: [{label_prob.min():.3f}, {label_prob.max():.3f}]")

        if has_label_sauna:
            label_sauna = item["label_sauna"][0].squeeze(0)
            print(f"  Label_sauna range: [{label_sauna.min():.3f}, {label_sauna.max():.3f}]")

        # 이미지 시각화를 위해 transpose
        if len(image.shape) == 2:
            cmap = "gray"
        elif len(image.shape) == 3:
            image = image.permute(1, 2, 0)
            cmap = None
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")

        col_idx = 0

        # 1. 이미지 표시
        im1 = axes[i, col_idx].imshow(image, cmap=cmap)
        axes[i, col_idx].set_title(f"Image {i+1}", fontsize=12, fontweight="bold")
        axes[i, col_idx].text(
            0.5,
            -0.1,
            f"file:{item['name'][0]}",
            fontsize=8,
            ha="center",
            va="top",
            transform=axes[i, col_idx].transAxes,
        )
        axes[i, col_idx].axis("off")
        if cmap:
            plt.colorbar(im1, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
        col_idx += 1

        # 2. 라벨 표시
        im2 = axes[i, col_idx].imshow(label, cmap="gray")
        axes[i, col_idx].set_title(f"Label {i+1}", fontsize=12, fontweight="bold")
        axes[i, col_idx].axis("off")
        plt.colorbar(im2, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
        col_idx += 1

        # 3. Label_prob 표시 (있는 경우)
        if has_label_prob:
            im3 = axes[i, col_idx].imshow(label_prob, cmap="viridis")
            axes[i, col_idx].set_title(f"Label_prob {i+1}", fontsize=12, fontweight="bold")
            axes[i, col_idx].axis("off")
            plt.colorbar(im3, ax=axes[i, col_idx], fraction=0.046, pad=0.04)
            col_idx += 1

        # 4. Label_sauna 표시 (있는 경우)
        if has_label_sauna:
            im4 = axes[i, col_idx].imshow(label_sauna, cmap="plasma")
            axes[i, col_idx].set_title(f"Label_sauna {i+1}", fontsize=12, fontweight="bold")
            axes[i, col_idx].axis("off")
            plt.colorbar(im4, ax=axes[i, col_idx], fraction=0.046, pad=0.04)

    fig.suptitle(dataset_name, fontsize=16, fontweight="bold")

    plt.tight_layout()
    # plt.subplots_adjust(top=0.9)
    plt.savefig(f"results/{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
