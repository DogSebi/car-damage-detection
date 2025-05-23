import numpy as np
import cv2


def analyze_damage(parts_masks, damage_mask, original_image, part_names=None):
    vis_image = original_image.copy()
    damage_report = []

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for i, part_mask in enumerate(parts_masks):
        part_area = (part_mask > 0.1).astype(np.uint8)
        if np.sum(part_area) == 0:
            continue

        overlap = cv2.bitwise_and(
            part_area, (damage_mask > 0).astype(np.uint8))

        damage_pixels = np.sum(overlap)
        part_pixels = np.sum(part_area)

        if damage_pixels == 0:
            continue

        damage_ratio = np.round(damage_pixels / part_pixels, 4)

        part_name = part_names[i] if part_names and i < len(
            part_names) else f"Part {i}"
        damage_report.append({
            "part": part_name,
            "damage_percent": damage_ratio
        })

        contours, _ = cv2.findContours(
            part_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, COLORS[i % len(COLORS)], 2)

        M = cv2.moments(part_area)

    if not damage_report and np.sum(damage_mask > 0) > 0:
        damage_report.append({
            "part": "Повреждение вне сегментированных частей автомобиля",
            "damage_percent": "Неизвестно",
        })

    red_mask = np.zeros_like(original_image)
    red_mask[:, :, 2] = (damage_mask > 0).astype(np.uint8) * 255
    cv2.addWeighted(red_mask, 0.4, vis_image, 0.6, 0, vis_image)

    return vis_image, damage_report
