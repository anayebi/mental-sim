import numpy as np
import torch
import torch.nn as nn
from ptutils.model_training.train_utils import (
    reduce_metric,
    AverageMeter,
    check_best_loss,
)
from mpmodels.model_training.video_trainer_base import BaseVideoTrainer


class SelfSupLossMinTrainer(BaseVideoTrainer):
    def __init__(self, config):
        super(SelfSupLossMinTrainer, self).__init__(config)

    def initialize_loss_function(self):
        # we will optimize through the model func
        return None

    def initialize_optimizer(self):
        assert hasattr(self, "config")
        assert hasattr(self, "model")
        self.check_key("optimizer_params")

        assert "initial_lr" in self.config["optimizer_params"].keys()

        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["optimizer_params"]["initial_lr"],
            weight_decay=self.config["optimizer_params"].get("weight_decay", 0.0),
        )
        return optim

    def train_one_epoch(self):
        assert hasattr(self, "train_loader")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")

        losses = AverageMeter("Loss", ":.4e")
        num_steps = len(self.train_loader)
        num_batches = len(self.train_loader)
        batch_size = (
            self.config["optimizer_params"]["train_batch_size"] // self.world_size
        )

        if self.use_tpu:
            import torch_xla.core.xla_model as xm

            tracker = xm.RateTracker()

        print("Setting model to train")
        self.set_model_to_train()
        for i, (data, labels) in enumerate(self.train_loader):
            curr_step = self.current_epoch * num_batches + i
            if not self.use_tpu:
                # For TPU, we have already assigned it to the device
                # use non_blocking: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/5
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward propagation
            loss = self.model(data)["loss"]

            # Backward propagation
            loss.backward()

            # Update parameters
            if self.use_tpu:
                xm.optimizer_step(self.optimizer)
                tracker.add(batch_size)
            else:
                self.optimizer.step()

            if self.use_tpu:
                rep_loss = loss.item()
            else:
                rep_loss = reduce_metric(loss, self.world_size).item()

            losses.update(rep_loss, data.size(0))

            if self.use_tpu:
                if curr_step % 10 == 0:
                    examples_seen = i * batch_size * self.world_size
                    examples_seen += (self.rank + 1) * batch_size
                    per_worker_header = (
                        f"[xla:{self.rank}, "
                        f"rate: {tracker.rate():.2f}, "
                        f"global_rate: {tracker.global_rate():.2f}]\t"
                    )
                    print(
                        f"{per_worker_header}"
                        f"Train Epoch: {self.current_epoch} "
                        f"[{examples_seen}/{len(self.train_loader._loader.dataset)} "
                        f"({100.0*i/num_batches:.0f}%)]"
                        f"\tLoss: {loss.item():.6f}"
                        f"\tStep: {curr_step}"
                    )
            elif self.rank == 0:
                print_str = (
                    f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                    f"Train Loss {rep_loss:.6f}"
                )
                self.print_fn(f"{print_str}")

        average_loss = losses.avg
        if self.use_tpu:
            # average across TPU replicas
            average_loss = xm.mesh_reduce("train_average_loss", average_loss, np.mean)

        # Print train results over entire dataset
        msg_str = "[Epoch {}] Train Loss: {:.6f}".format(
            self.current_epoch, average_loss
        )
        if self.use_tpu:
            # xm.master_print only prints on the first TPU core
            self.print_fn(msg_str)
        elif self.rank == 0:
            # print all reduce result on one gpu
            self.print_fn(msg_str)

        self.results["losses"]["train"].append(average_loss)

    def validate(self):
        assert hasattr(self, "val_loader")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")

        if (self.current_epoch % self.config["save_freq"] == 0) or (
            self.current_epoch + 1 == self.config["num_epochs"]
        ):
            losses = AverageMeter("Loss", ":.4e")
            num_steps = len(self.val_loader)

            self.set_model_to_eval()
            with torch.no_grad():
                for i, (data, labels) in enumerate(self.val_loader):
                    if not self.use_tpu:
                        # For TPU, we have already assigned it to the device
                        # use non_blocking: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/5
                        data = data.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                    loss = self.model(data)["loss"]

                    if self.use_tpu:
                        rep_loss = loss.item()
                    else:
                        rep_loss = reduce_metric(loss, self.world_size).item()

                    losses.update(rep_loss, data.size(0))

                    if (not self.use_tpu) and (self.rank == 0):
                        print_str = (
                            f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                            f"Val Loss {rep_loss:.6f}"
                        )
                        self.print_fn(f"{print_str}")

            average_loss = losses.avg
            if self.use_tpu:
                # Average across TPU replicas
                import torch_xla.core.xla_model as xm

                average_loss = xm.mesh_reduce("val_average_loss", average_loss, np.mean)

            # Print val results over entire dataset
            msg_str = "[Epoch {}] Val Loss: {:.6f}".format(
                self.current_epoch, average_loss
            )
            if self.use_tpu:
                # xm.master_print only prints on the first TPU core
                self.print_fn(msg_str)
            elif self.rank == 0:
                # print all reduce result on one gpu
                self.print_fn(msg_str)

            self.results["losses"]["val"].append(average_loss)

            # Check if current loss is best
            self.best_loss, self.is_best = check_best_loss(average_loss, self.best_loss)
