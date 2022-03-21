
from gettext import find
import math

def active_smapling():
    return

def query_weak_annotations():
    return

#Calculates distance from the bounding box's center to the position of the weak annotation
def dist_from_point(box, weak_annotation):
    box_x1 = box[0]
    box_x2 = box[2]
    box_y1 = box[1]
    box_y2 = box[3]
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1
    box_center = (box_x1 + box_width/2.0, box_y1 + box_height/2.0)
    dist = math.sqrt(math.pow(box_center[0] - weak_annotation[0], 2) + math.pow(box_center[1] - weak_annotation[1], 2))
    return dist

#finds the bounding box with its center closest to the weak annotation
def find_closest_box(bounding_boxes, scores, classes, annotation):
    closest_box = ()
    closest_distance = float('inf')
    for box, score, c in zip(bounding_boxes, scores, classes):
        dist = dist_from_point(box, annotation)
        if dist < closest_distance:
            closest_box = (box, score, c)
            closest_distance = dist
    return closest_box

def pseudo_labels(model, sample, weak_annotations):
    #predict bounding boxes
    #use weak labels to choose best possible bounding box
    #   - for every click location, we pseudo label that object with a
    #     bounding-box with center closest to the click location.
    #   - The object is classified as the class with the
    #     highest probability for the chosen bounding box
    #   - For each image we calculate the confidence score wich is the mean score
    #     which is the mean probability score obtained for each predicted object
    labels_and_confidence = []
    for i, (image, annotations) in enumerate(zip(sample, weak_annotations)):
        pseudo_labels = []
        boxes, scores, classes, _ = model.predict(image)
        for annotation in annotations:
            closest_box = find_closest_box(boxes[0], scores[0], classes[0], annotation)
            pseudo_labels.append(closest_box)
        confidence_score = 0

        for label in pseudo_labels:
            confidence_score += label[1]
        confidence_score = confidence_score/len(pseudo_labels)
        labels_and_confidence.append((pseudo_labels, confidence_score, i))

    return labels_and_confidence

def query_annotations():
    return

def soft_switch(pseudo_labels, conf_thresh):
    s_high = []
    s_low = []
    for sample in pseudo_labels:
        confidence = sample[1]
        if confidence > conf_thresh:
            s_high.append(sample)
        else:
            s_low.append(sample)

def adaptive_supervision(unlabeled_pool, labeled_pool, weak_labeled_pool, model, episode_num, sample_size, soft_switch_thresh):
    s = active_smapling()
    w_s = query_weak_annotations()
    p_s = pseudo_labels()

    return model