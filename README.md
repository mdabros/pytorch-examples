pytorch-examples
=====================

Repository for pytorch examples and experiments.

## Examples

 * [**1_cifar10_model_architectures**](src/1_cifar10_network_architectures.py): Trains the selected network architecture on the CIFAR10 dataset.
 * [**2_pytorch_hyperparameter_tuning_with_ray**](src/2_pytorch_hyperparameter_tuning_with_ray.py): Search for optimizer hyperparameters using Ray on the CIFAR10 dataset.
 
## Installation

### Python 3.6 in Visual Studio 2019
Use **Visual Studio Installer** -> select **Modify**.

Workload:
 * Install **Data science and analytical applications**  workload

Components:
 * Install **Python language support**
 * Install **Python 3 64-bit** (e.g. 3.6.3)

Install required packages:
 * Start Visual Studio, open **Python Environments**
 * Select **Python 3.6 (64-bit)**
 * On the **Overview** combo box select **Install from requirements.txt**
 
### Installing cuDNN
For the gpu version, make sure to check which version of cuda and cuDNN is supported. 
Then follow the instructions on [cudnn-install-windows](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows)

