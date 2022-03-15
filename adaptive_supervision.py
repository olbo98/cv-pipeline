
def active_smapling():
    return

def query_weak_annotations():
    return

def pseudo_labels():
    return

def query_annotations():
    return

def adaptive_supervision(unlabeled_pool, labeled_pool, weak_labeled_pool, model, episode_num, sample_size, soft_switch_thresh):
    s = active_smapling()
    w_s = query_weak_annotations()
    p_s = pseudo_labels()

    return model