from catalyst.contrib.utils import plot_tensorboard_log
from model import Model
import torch.nn as nn


class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        # model train/valid step
        features, targets = batch["features"], batch["targets"]
        logits = self.model(features)
        scores = torch.sigmoid(logits)

        loss = self.criterion(logits, targets)
        accuracy, precision, recall, iou = multi_label_metrics(
            logits, targets, threshold=0.5, activation="Sigmoid"
        )
        map05, map10, map20 = mean_average_precision_at_k(
            scores, targets, top_k=(5, 10, 20)
        )
        batch_metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "map05": map05,
            "map10": map10,
            "map20": map20
        }

        self.input = {"features": features, "targets": targets}
        self.output = {"logits": logits, "scores": scores}
        self.batch_metrics.update(batch_metrics)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict_batch(self, batch):
        # model inference step
        batch = utils.maybe_recursive_call(batch, "to", device=self.device)
        logits = self.model(batch["features"])
        scores = torch.sigmoid(logits)
        return scores

if __name__ == '__main__':
    check = True

    if check:
        model = Model()
        criterion = nn.BCEWithLogitsLoss()
        batch = next(iter(train_loader))
        output = model(batch['features'])
        loss = criterion(output, batch['targets'])
        print(loss)

    model = Model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loaders = {"train": train_loader, "valid": valid_loader}

    runner_test = CustomRunner()

    runner_test.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        loaders=loaders,
        logdir="./logs",
        num_epochs=10,
        verbose=True,
        load_best_on_end=True,
        overfit=False,  # <<<--- DO NOT FORGET TO MAKE IT ``False``
        #  (``True`` uses only one batch to check pipeline correctness)
        callbacks=[
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
            # dl.AveragePrecisionCallback(input_key="targets", output_key="scores", prefix="ap"),
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
            # dl.AUCCallback(input_key="targets", output_key="scores", prefix="auc"),
        ],
        main_metric="iou",  # "ap/mean",
        minimize_metric=False,
    )

    # model inference example
    for prediction in runner.predict_loader(loader=loaders["valid"]):
        assert prediction.detach().cpu().numpy().shape[-1] == MERCH_TYPE_NCLASSES - 1

    plot_tensorboard_log(
        logdir="./logs",
        step="epoch",
        metrics=[
            "loss", "accuracy", "precision", "recall", "iou",
            "map05", "map10", "map20",
            "ap/mean", "auc/mean"
        ]
    )