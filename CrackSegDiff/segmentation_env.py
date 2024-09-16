import sys
sys.path.append(".")
import numpy as np
import math
import torch
import torchvision
from PIL import Image
import argparse
import os
import cv2
major = cv2.__version__.split('.')[0]     # Get opencv version
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, jaccard_score
def calculate_metrics(pred, gt, pred_, gt_):
    pred_ = np.array(pred_)
    gt_ = np.array(gt_)
    return (f1_score(gt.reshape(-1), pred.reshape(-1), zero_division=0),
            precision_score(gt.reshape(-1), pred.reshape(-1), zero_division=0),
            recall_score(gt.reshape(-1), pred.reshape(-1), zero_division=0),
            accuracy_score(gt.reshape(-1), pred.reshape(-1)),
            jaccard_score(gt.reshape(-1), pred.reshape(-1), zero_division=0),
            bfscore(pred_, gt_, 3))
def bfscore(prfile, gtfile, threshold):

    gt_ = gtfile    # Convert color space
    pr_ = prfile    # Convert color space
    classes_gt = np.unique(gt_)    # Get GT classes
    classes_pr = np.unique(pr_)    # Get predicted classes
    # Check classes from GT and prediction
    if not np.array_equiv(classes_gt, classes_pr):
        classes = np.concatenate((classes_gt, classes_pr))
        classes = np.unique(classes)
        classes = np.sort(classes)
    else:
        classes = classes_gt    # Get matched classes
    m = int(np.max(classes))    # Get max of classes (number of classes)
    # Define bfscore variable (initialized with zeros)
    bfscores = np.zeros((m+1), dtype=float)
    areas_gt = np.zeros((m+1), dtype=float)
    for i in range(m+1):
        bfscores[i] = np.nan
        areas_gt[i] = np.nan
    for target_class in classes:    # Iterate over classes
        if target_class == 0:     # Skip background
            continue
        gt = gt_.copy()
        gt[gt != target_class] = 0
        # contours는 point의 list형태.
        if major == '3':    # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    # Find contours of the shape
        else:    # For other opencv versions
            contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape

        # contours 는 list of numpy arrays
        contours_gt = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_gt.append(contours[i][j][0].tolist())
        # Get contour area of GT
        if contours_gt:
            area = cv2.contourArea(np.array(contours_gt))
            areas_gt[target_class] = area
        pr = pr_.copy()
        pr[pr != target_class] = 0
        # contours는 point의 list형태.
        if major == '3':    # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:    # For other opencv versions
            contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # contours 는 list of numpy arrays
        contours_pr = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_pr.append(contours[i][j][0].tolist())
        # 3. calculate
        if len(contours) == 0:
            return 0
        else:
            precision, numerator, denominator = calc_precision_recall(
            contours_gt, contours_pr, threshold)    # Precision

            recall, numerator, denominator = calc_precision_recall(
            contours_pr, contours_gt, threshold)    # Recall


        if (recall + precision) == 0:
            f1 = 0
        else:
            f1 = 2*recall*precision/(recall + precision)    # F1 score
        bfscores[target_class] = f1


    # return bfscores[1:], np.sum(bfscores[1:])/len(classes[1:])    # Return bfscores, except for background, and per image score
    return bfscores[-1]   # Return bfscores, except for background
def calc_precision_recall(contours_a, contours_b, threshold):
    x = contours_a
    y = contours_b

    xx = np.array(x)
    hits = []

    for yrec in y:
        d = np.square(xx[:,0] - yrec[0]) + np.square(xx[:,1] - yrec[1])

        hits.append(np.any(d < threshold*threshold))
    top_count = np.sum(hits)

    try:
        precision_recall = top_count / len(y)
    except ZeroDivisionError:
        precision_recall = 0

    return precision_recall, top_count, len(y)

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--inp_pth")
    argParser.add_argument("--out_pth")
    args = argParser.parse_args()
    mix_res = (0., 0., 0., 0., 0., 0.)
    num = 0
    pred_path = args.inp_pth
    gt_path = args.out_pth
    for root, dirs, files in os.walk(pred_path, topdown=False):
        for name in files:
            if 'ens' in name:
                num += 1
                ind = name.split('_')[0]
                pred_ = Image.open(os.path.join(root, name)).convert('L')
                gt_name = ind + ".bmp"
                gt_ = Image.open(os.path.join(gt_path, gt_name)).convert('L')
                pred = torchvision.transforms.PILToTensor()(pred_)
                pred = torch.unsqueeze(pred,0).float()
                if pred.max() == 0:
                    pred = pred
                else:
                    pred = pred / pred.max()
                gt = torchvision.transforms.PILToTensor()(gt_)
                gt = torch.unsqueeze(gt, 0).float() / 255.0
                temp = calculate_metrics(pred, gt, pred_, gt_)
                printed = False
                for x in temp :
                    if x < 0.7 and not printed:
                        print(ind,temp[0],temp[4],temp[5])
                        printed = True
                        break
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
    F1, Precision, Recall, Accuracy, IoU, BFScore= tuple([a / num for a in mix_res])
    print('IoU is', IoU)
    print('F1 is', F1)
    print('Precision is', Precision)
    print('Recall is', Recall)
    print('Accuracy is', Accuracy)
    print('BFScore is', BFScore)
if __name__ == "__main__":
    main()
