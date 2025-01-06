import os
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only


class CheckpointEveryNSteps(pl.Callback):
    def __init__(
        self,
        checkpoints_dir,
        save_step_frequency,
    ) -> None:
        r"""Save a checkpoint every N steps.

        Args:
            checkpoints_dir (str): directory to save checkpoints
            save_step_frequency (int): save checkpoint every N step
        """

        self.checkpoints_dir = checkpoints_dir
        self.save_step_frequency = save_step_frequency

    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs) -> None:
        r"""Save a checkpoint every N steps."""

        trainer = args[0]
        global_step = trainer.global_step

        if global_step == 1 or global_step % self.save_step_frequency == 0:

            ckpt_path = os.path.join(
                self.checkpoints_dir,
                "step={}.ckpt".format(global_step))
            trainer.save_checkpoint(ckpt_path)
            print("Save checkpoint to {}".format(ckpt_path))


'''class GradientLoggingCallback(pl.Callback):
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        step_count = trainer.global_step

        # 100 스텝마다 기울기 기록
        if step_count % self.log_every_n_steps == 0:
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    wandb.log({f"grad_{name}_histogram": wandb.Histogram(param.grad.cpu().numpy())})'''

