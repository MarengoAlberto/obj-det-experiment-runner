import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_bbox(image, boxes, labels=None, conf_scores = None, color=(255, 0, 0), thickness=-1):

    for idx, box in enumerate(boxes):
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]

        image = np.ascontiguousarray(image)
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

        if labels:
            display_text = str(labels[idx])

            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

            cv2.rectangle(image, (xmin, ymin - int(0.9 * text_height)), (xmin + int(0.4*text_width), ymin), color, -1)


            image = cv2.putText(
                image,
                display_text,
                (xmin, ymin - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        elif conf_scores:
            display_text = str(round(conf_scores[idx], 4)*100) + '%'

            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

            cv2.rectangle(image, (xmin, ymin - int(0.9 * text_height)), (xmin + int(0.4*text_width), ymin), color, -1)


            image = cv2.putText(
                image,
                display_text,
                (xmin, ymin - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

    return image

def plot_predictions(prediction):
    # Define subplots
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 10),
    )
    # Plot Predictions.
    orig_image = draw_bbox(
        prediction['image_array'],
        np.array(prediction['final_boxes'], dtype=np.int32),
        labels=prediction['class_names'],
        color=(255, 0, 0),
        thickness=2,
    )


    ax.imshow(orig_image)
    ax.axis("off")

    plt.figtext(0.50, 0.9, "Predictions", fontsize=20, color="r", ha="center")

    plt.show(block=False)

    return fig
