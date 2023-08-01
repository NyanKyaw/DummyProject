import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=11):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum") # remember all the losses have some form of squared error

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 12:16], target[..., 12:16])
        iou_b2 = intersection_over_union(predictions[..., 17:21], target[..., 12:16])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two predictions
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 11].unsqueeze(3)  # Does a box exist? 0 for no, 1 for yes

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 17:21] # if second bounding box is bext (i.e bestbox = 1)
                + (1 - bestbox) * predictions[..., 12:16] # if first bounding box is best (i.e bestbox = 0)
            )
        )

        box_targets = exists_box * target[..., 12:16]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt( # we take the sign as taking the abs value removes it
            torch.abs(box_predictions[..., 2:4] + 1e-6) # we take the sqrt of the abs value (can't have negative, add small value so its not 0)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 17:18] + (1 - bestbox) * predictions[..., 11:12] # remember, its the probability of an object that we want
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box), # remember that we only do this if there are objects (i.e there is a ground truth box, exist_box = 1)
            torch.flatten(exists_box * target[..., 11:12]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # We do this for both boxes in our interpretation! Remember only do this if there is no objects (i.e no ground truth box, exist_box = 0)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 11:12], start_dim=1), # remember, its the probability of an object that we want
            torch.flatten((1 - exists_box) * target[..., 11:12], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 17:18], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 11:12], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :11], end_dim=-2,),
            torch.flatten(exists_box * target[..., :11], end_dim=-2,),
        )

        # ================== #
        #    TOTAL LOSS      #
        # ================== #
        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss