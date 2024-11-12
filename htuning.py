from lightorch.hparams import htuning
from typing import List, Dict
from .data import DataModule
from .loss import criterion
from .model import Model
import optuna

labels: List[str] | str = criterion().labels  #


def objective(trial: optuna.trial.Trial) -> Dict:
    num_layers: int = trial.suggest_int("num_layers", 1, 5)
    return dict(
        optimizer="adam",
        scheduler="one-cycle",
        layers=[
            trial.suggest_int(f"layer_{i}", 1, 128) for i in range(1, num_layers + 1)
        ],
        activations=[
            trial.suggest_categorical(f"activation_{i}", ["ReLU", "Softmax"])
            for i in range(1, num_layers + 1)
        ],
        lr=trial.suggest_float("lr", 1e-5, 5e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    )


if __name__ == "__main__":
    htuning(
        model_class=Model,
        hparam_objective=objective,
        datamodule=DataModule,
        valid_metrics=labels,
        datamodule_kwargs=dict(pin_memory=True, num_workers=8, batch_size=32),
        directions=["minimize" for _ in range(len(labels))],
        precision="high",
        n_trials=10000,
        trainer_kwargs=dict(
            logger=True,
            enable_checkpointing=False,
            max_epochs=10,
            accelerator="cuda",
            devices=1,
            log_every_n_steps=22,
            precision="32",
            limit_train_batches=1 / 3,
            limit_val_batches=1 / 3,
        ),
    )
