import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from ibm.models import AddConstant


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Instantiate the model using the config
    model = AddConstant(c=cfg.model.c)

    # Initialize Weights & Biases run
    with wandb.init(
        project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True)
    ) as run:

        # Mock training loop
        for i in range(cfg.iterations):

            x = torch.randn(1)
            pred = model(x)

            if i % cfg.log_interval == 0:
                print(f"Iteration {i}, Input: {x.item()}, Prediction: {pred.item()}")

                # Log predictions to Weights & Biases
                run.log({"prediction": pred.item()}, step=i)


if __name__ == "__main__":
    main()
