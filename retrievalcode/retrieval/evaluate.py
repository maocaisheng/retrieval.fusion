def read_csv(fname, sep):
    with open(fname, 'r') as f:
        result = {}
        for line in f:
            splits = line.strip().split(sep)
            result[splits[0]] = splits[1:]
    return result


def comp_ap(ranked_list, gt):
    recall = 0
    precision = 0
    # remove duplicate elements
    gt = set(gt)
    ranked_list = sorted(set(ranked_list), key=ranked_list.index)
    for idx, item in enumerate(ranked_list, 1):
        if item in gt:
            recall += 1
            precision += recall / idx
    if recall != 0:
        ap = precision / recall
        r = recall / len(gt)
        p = recall / len(ranked_list)
    else:
        ap = 0
        r = 0
        p = 0
    return ap, r, p


def comp_mAP(query, gt, verbose):  # dic{query_name: rank_list}
    mAP = 0
    mR = 0
    mP = 0
    for qn in query.keys():
        ap, r, p = comp_ap(query[qn], gt[qn])
        mAP += ap
        mR += r
        mP += p
        if verbose:
            print('%s, ap: %.3f, recall: %.3f, precision:%.3f' % (qn, ap, r, p))
    qnum = len(query)
    mAP /= qnum
    mR /= qnum
    mP /= qnum
    return mAP, mR, mP


def evaluate(query_file, gt_file, topN, verbose=False, sep=' '):
    query = read_csv(query_file, sep)
    gt = read_csv(gt_file, sep)
    mAP, mR, mP = comp_mAP(query, gt, verbose)
    print('~~~~~~~~~~~~~~~~~~~~~')
    print('mAP@%d        =%.4f\n' % (topN, mAP))
    print('mean Recall   =%.4f\n' % (mR))
    print('mean Precision=%.4f\n' % (mP))


if __name__ == '__main__':
    mAP, mR, mP = comp_mAP(query={1: list(range(1, 11)), 2: list(range(1, 11))}, gt={1: [1, 3, 6, 9, 10], 2: [2, 5, 7]},
                           verbose=False)
    print('mAP@%d=%.2f' % (10, mAP))
    print('mean Recall=%.2f' % (mR))
    print('mean Precision=%.2f' % (mP))