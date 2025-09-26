# Best practices for ML codebases

Principles we want to adhere to:
- **readable** :eyes:
- **runnable** :arrow_forward:	
- **trackable** :chart_with_upwards_trend:
- **reproducible** :repeat:


## Code readability

Clean code is easy to read, understand and to extend over time. Investing in it early prevents technical debt when the project complexity increases.

### Project structure

Readability starts with the project structure. It should be easy to figure out where to find source files, scripts, configs, etc. Different components should be clearly delimited. Start with the following structure:

```
├── config              [configuration files]
├── data                [raw datasets, optional here]
├── notebooks           [self-contained examples]
├── scripts             [high-level script files] 
├── src                 [source files implementing the actual logic]
    └── [project_name]  [define the project package]
        └── ...         [all source files are nested here]
├── tests               [testing files]
└── README.md           [describes the project and how to run it]

```

> :bulb: **Tip:** You can use this [project template](https://gitlab.ethz.ch/image-based-mapping-hs2025/general/project-template).

### Formatting
Always keep code formatted according to a [coding standard](https://peps.python.org/pep-0008/). To simplify this process, use an automatic code formatter, such as [Black](https://github.com/psf/black).

> :bulb: **Tip:** You can [automate](https://dev.to/emmo00/how-to-setup-black-and-pre-commit-in-python-for-auto-text-formatting-on-commit-4kka) code formatting before commits.


### Type annotation

Python is flexbile and doesn't enforce type annotation. However, it is good practice to use [type hints](https://docs.python.org/3/library/typing.html) for your function variables.

```python

# Without type-hints
def get_character_count(name):
    return len(name)

# With type-hints
def get_character_count(name: str) -> int:
    return len(name)
```

> :warning: Skipping type hints could create the impression of a more generic implementation, but most of the times this is not needed and the resulting ambiguity is error-prone. If you really need to support multiple input types, use [Union](https://docs.python.org/3/library/typing.html#typing.Union) to define that.  

For example, see the `activation` parameter of the `Transformer` module's [implementation in PyTorch](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L107):

```python
class Transformer(Module):
    def __init__(
        self,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        # Some other parameters ...
    )
```

The parameter `activation` can be either a `str` like `"relu"` or `"gelu"` or a function (`Callable`) that receives as input a `Tensor` and also outputs a `Tensor`.

### Docstrings

Generally, good naming of the functions and their parameters is enough to understand how to call them, but this can be improved with a more detailed documentation in the form of [docstrings](https://peps.python.org/pep-0257/). Common standards are the [Google](https://google.github.io/styleguide/pyguide.html) and the [NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) styles.

**TODO example**

> :bulb: **Tip:** You can automate the creation of docstrings in your preferred style with VSCode extensions such as [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).

### Design patterns
A good design of the codebase also allows easier experimentation by swapping components, for instance during ablation studies. Explore [single-responsability-principle](https://www.youtube.com/watch?v=pTB30aXS77U), [dependency-inversion](https://www.youtube.com/watch?v=2ejbLVkCndI) and the [strategy-pattern](https://www.youtube.com/watch?v=n2b_Cxh20Fw).

## Running

To make your code reproducible, it first needs to be runnable by other people. 

### Virtual environment

A clean way to work in Python is by using virtual environments. This solves the problem of global installations and conflicting dependencies between projects. There are multiple tools you can choose from: [conda](https://www.anaconda.com/docs/getting-started/working-with-conda/environments), [venv](https://docs.python.org/3/library/venv.html), [uv](https://docs.astral.sh/uv/pip/environments/), etc. Install and use your preferred one.

For example, to create an environment using conda, run this command:
```bash
# Create an environment called `ibm` with the python version 3.10
conda create -n ibm python=3.10

# Activate the environment
conda activate ibm
```

It is good-practice to define a set of dependencies to be installed in your environment in the form of a `requirements.txt` file:

```python
# Example requirements.txt file
matplotlib
scipy
torch==2.4.1
torchvision==0.19.1
transformers>=4.32.1
```

After creating and activating the new environment, all the requirements can be installed with the following command:

```bash
pip install -r requirements.txt
```

> :warning: This procedure generally covers most use cases. If your project requires a more complex installation procedure, clearly describe it in the README.

### Configuration manager

Don't use constants in your codebase for the hyperparameters of the method. You should be able to pass them from the command line when launching experiments or running evaluations.

### Argparse

Using the `argparse` standard library you can define a set of arguments for your program that you can set from the command line. Create the file `argparse_example.py`:

```python
import argparse
    
def main():
    # Definition
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)

    # Parse actual CLI arguments 
    args = parser.parse_args()

    print("Epochs:", args.foo)

if __name__ == "__main__":
    main()
```

Run the script from the CLI and pass the desired argument:

```bash
python argparse_example.py --epochs=10
# Epochs: 10
```

### Hydra / OmegaConf
While argparse allows CLI overrides, a more generic approach involves hierarchical configs in a YAML format. Hydra is a framework based on [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/index.html) that allows composing and overriding config files.

Create a config file `configs/example_config.yaml`:
```yaml
trainer:
    epochs: 10

model:
    layers: 5
```

Integrate in a Python script called `hydra_example.py`:

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="example_config")
def main(cfg : DictConfig) -> None:
    print("Epochs:", cfg.trainer.epochs)
    print("Layers:", cfg.model.layers)

if __name__ == "__main__":
    main()
```

Run it and override one value of the YAML config:
```bash
python example_hydra.py trainer.layers=3
# Epochs: 10
# Layers: 3
```

> :rocket: Hydra has more complex features, including [composition](https://hydra.cc/docs/intro/#composition-example). **Explore the documentation for more [tutorials](https://hydra.cc/docs/tutorials/intro/).**

> :bulb: **Tip:** It is possible to define all the components in a nested config and use Hydra to [recursively instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/) objects in one line.



## Tracking

To asses the performance of a ML model, we measure training and evaluation metrics such as loss, predictions and performance scores scores at different iterations. To track and visualize metrics, use a tool like [Weights & Biases](https://wandb.ai/site/) or [Tensorboard](https://www.tensorflow.org/tensorboard).

### Weights & Biases

WandB is a popular choice due to its ease of use and cloud availability. It can log metrics, images and other artifacts like code or even checkpoints.

Read the [quickstart guide](https://docs.wandb.ai/quickstart/) and explore the tutorials for [tracking experiments](https://docs.wandb.ai/tutorials/experiments/).

## Reproducibility

Following all the previous guidelines sets a good foundation for a reproducible pipeline (i.e. defining environment requirements, providing configuration files for the experiments and exposing scripts to run them.)

### Random seed

To minimize variability, set the seed for all pseudorandom number generators using a function like the one bellow: 

```python
import random
import numpy as np
import torch

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

### README

A README file should encapsulate all the steps needed to reproduce the experiments: how to install the environment, how to prepare the data, how to run training and evaluation, etc. Follow the [structure](https://gitlab.ethz.ch/image-based-mapping-hs2025/general/ibm-project-template/-/blob/main/README.md?ref_type=heads) from the project template.


## TLDR
- Use a clear project structure to separate source files, scripts, configs, etc.

- Keep code formated, use Black.

- Write docstrings for every class or function.

- Use best practices for coding design (modularity, single responsability principle, etc.).

- Use virtual environments (Anaconda, uv, etc.)

- Use configuration files and a configuration manager (Omegaconf or Hydra).

- Create self-contained scripts or notebooks as high-level entry points (launch training, evaluation, etc.).

- Track metrics using WandB or Tensorboard.

- Create README with clear instructions how to reproduce your experiments.
