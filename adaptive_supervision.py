
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
def find_closest_box(bounding_boxes, annotation):
    closest_box = []
    closest_distance = float('inf')
    for box in bounding_boxes:
        dist = dist_from_point(box, annotation)
        if dist < closest_distance:
            closest_box = box
            closest_distance = dist
    return closest_box

def pseudo_labels(model, sample, weak_annotations):
    #predict bounding boxes
    #use weak labels to choose best possible bounding box
    #   - for every click location, we pseudo label that object with a
    #     bounding-box with center closest to the click location.
    #   - The object is classified as the class with the
    #     highest probability for the chosen bounding box
    #TODO: Need to return confidence scores. Also the classes need to be paired with the boxes
    chosen_bounding_boxes = []
    for image, annotations in zip(sample, weak_annotations):
        boxes, scores, classes, _ = model.predict(image)
        for annotation in annotations:
            closest_box = find_closest_box(boxes[0], annotation)
            chosen_bounding_boxes.append(closest_box)
    return chosen_bounding_boxes

def query_annotations():
    return

def adaptive_supervision(unlabeled_pool, labeled_pool, weak_labeled_pool, model, episode_num, sample_size, soft_switch_thresh):
    s = active_smapling()
    w_s = query_weak_annotations()
    p_s = pseudo_labels()

    return model