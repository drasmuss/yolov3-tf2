import timeit

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.core.standard_fields import InputDataFields, DetectionResultFields

from yolov3_tf2.models import YoloV3
from yolov3_tf2.utils import draw_outputs


data, info = tfds.load("coco", with_info=True, data_dir="./data")

n_classes = info.features["objects"]["label"].num_classes
# n_samples = info.splits["validation"].num_examples
n_samples = 2048
minibatch_size = 32
img_size = 416

yolo = YoloV3(size=img_size, classes=n_classes, iou_threshold=0.5, score_threshold=0.5)

yolo.load_weights("./checkpoints/yolov3.tf")

with open("./data/coco/2014/1.1.0/objects-label.labels.txt") as f:
    class_names = [c.strip() for c in f.readlines()]

assert len(class_names) == n_classes

validation_data = (
    data["validation"]
    .map(
        lambda feat: tf.image.resize(feat["image"], (img_size, img_size)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    .batch(minibatch_size, drop_remainder=True)
    .map(lambda img: img / 255, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# start = timeit.default_timer()
# for i, _ in enumerate(validation_data.take(100)):
#     print(i)
# print(timeit.default_timer() - start)

print("data")
print(info)
print(info.features)
print(info.splits)
print(validation_data)

boxes, scores, classes, nums = yolo.predict(
    validation_data, verbose=1, steps=n_samples // minibatch_size
)

# for img in validation_data.take(n_samples // minibatch_size):
#     img = img.numpy()
#
#     start = timeit.default_timer()
#     boxes, scores, classes, nums = yolo.predict(img)
#     print(timeit.default_timer() - start)

print("output")
print(boxes.shape)
print(scores.shape)
print(classes.shape)
print(nums.shape)

# compute map
evaluator = CocoDetectionEvaluator(
    [{"id": i, "name": n} for i, n in enumerate(class_names)]
)
for i, feat in enumerate(data["validation"].take(n_samples)):
    img_id = feat["image/id"].numpy()
    objects = feat["objects"]

    ground_truth = {
        InputDataFields.groundtruth_boxes: objects["bbox"].numpy(),
        InputDataFields.groundtruth_classes: objects["label"].numpy(),
        InputDataFields.groundtruth_is_crowd: objects["is_crowd"].numpy(),
    }

    # print("ground truth")
    # print(ground_truth)

    evaluator.add_single_ground_truth_image_info(img_id, ground_truth)

    # height, width = feat["image"].shape[:2]
    detection_results = {
        # note: need to swap bboxes from (x, y, x, y) to (y, x, y, x)
        DetectionResultFields.detection_boxes: boxes[i, : nums[i]][..., [1, 0, 3, 2]],
        DetectionResultFields.detection_scores: scores[i, : nums[i]],
        DetectionResultFields.detection_classes: classes[i, : nums[i]],
    }

    # print("detection results")
    # print(detection_results)

    evaluator.add_single_detected_image_info(img_id, detection_results)

results = evaluator.evaluate()

for i, img in enumerate(validation_data.unbatch().take(5)):
    img = img.numpy()
    print("example", i)
    print("detections:")
    for j in range(nums[i]):
        print(
            "\t{}, {}, {}".format(
                class_names[int(classes[i][j])],
                np.array(scores[i][j]),
                np.array(boxes[i][j]),
            )
        )
    img = draw_outputs(
        img, (boxes[[i]], scores[[i]], classes[[i]], nums[[i]]), class_names
    )

    plt.figure()
    plt.imshow(img)
    plt.savefig("./data/output/image_%d" % i)
