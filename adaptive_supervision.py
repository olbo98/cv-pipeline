
import math

def active_smapling():
    return

def query_weak_annotations():
    return

def dist_from_point(box, weak_annotation):
    box_x1 = box[0]
    box_x2 = box[2]
    box_y1 = box[1]
    box_y2 = box[3]
    box_center = (math.abs(box_x2 - box_x1)/2, math.abs(box_y1 - box_y2)/2)
    dist = math.sqrt(math.pow(box_center[0] - weak_annotation[0], 2) + math.pow(box_center[1] - weak_annotation[1], 2))
    return dist

def pseudo_labels(model, sample, weak_annotations):
    for image, weak_annotation in zip(sample, weak_annotations):
        boxes, scores, classes, nums = model.predict(image)


    #predict bounding boxes
    #use weak labels to choose best possible bounding box
    #   - for every click location, we pseudo label that object with a
    #     bounding-box with center closest to the click location.
    #   - The object is classified as the class with the
    #     highest probability for the chosen bounding box
    return

def query_annotations():
    return

def adaptive_supervision(unlabeled_pool, labeled_pool, weak_labeled_pool, model, episode_num, sample_size, soft_switch_thresh):
    s = active_smapling()
    w_s = query_weak_annotations()
    p_s = pseudo_labels()

    return model