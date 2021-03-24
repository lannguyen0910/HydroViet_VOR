from torch._C import device
from .baseline import BaselineModel
import torch
import models.object_detection.ssd.model as ssd


class Detector(BaselineModel):
    def __init__(self, n_categories, **kwargs):
        super(Detector, self).__init__(n_classes=n_categories, **kwargs)
        self.model = ssd.SSD300(n_classes=n_categories)
        self.model_name = "SSD300"
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)
        self.criterion = self.criterion(self.model.priors_cxcy)
        self.n_categories = n_categories

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
            self.criterion.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch['imgs']
        boxes = batch['boxes']
        categories = batch['categories']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            categories = [x.to(self.device) for x in categories]

        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, categories)
        return loss

    def inference_step(self, batch):
        inputs = batch['imgs']
        boxes = batch['boxes']
        categories = batch['categories']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            categories = [x.to(self.device) for x in categories]

        outputs = self(inputs)

        if self.device:
            outputs = [i.cpu().numpy() for i in outputs]

        return outputs

    def evaluate_step(self, batch):
        inputs = batch['imgs']
        boxes = batch['boxes']
        categories = batch['categories']

        if self.device:
            inputs = inputs.to(self.device)
            boxes = [x.to(self.device) for x in boxes]
            categories = [x.to(self.device) for x in categories]

        loc_preds, cls_preds = self(inputs)
        loss = self.criterion(loc_preds, cls_preds, boxes, categories)

        det_boxes, det_categories, det_scores = self.model.detect_objects(
            loc_preds, cls_preds, min_score=0.01, max_overlap=0.45, top_k=200
        )

        metric_dict = self.update_metrics(
            outputs={
                'det_boxes': det_boxes,
                'det_categories': det_categories,
                'det_scores': det_scores
            },

            targets={
                'gt_boxes': boxes,
                'gt_categories': categories
            }
        )
        return loss, metric_dict

    def forward_test(self):
        inputs = torch.rand(1, 3, 300, 300)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)

        return outputs
