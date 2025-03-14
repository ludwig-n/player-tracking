import ultralytics.engine.model
import wandb


def train_detector(
    model: ultralytics.engine.model.Model,
    wandb_key: str | None,
    **train_kwargs
) -> None:
    """
    Trains a detector model, optionally with W&B logging
    (set `wandb_key=None` to disable).
    See Ultralytics docs for supported training arguments.
    """
    if wandb_key is not None:
        wandb.login(anonymous="never", key=wandb_key)
    ultralytics.settings.update({"wandb": wandb_key is not None})
    model.train(**train_kwargs)


def eval_detector(
    model: ultralytics.engine.model.Model, dataset_yaml_path: str
) -> dict[str, float]:
    """
    Evaluates a detector model on a dataset.
    Ultralytics does this automatically after training,
    so this is useful mostly to evaluate pretrained models.
    Returns a results dict with metrics.
    """
    return model.val(data=dataset_yaml_path).results_dict
