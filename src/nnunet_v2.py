# %% [code]

import warnings

import joblib
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import glob
import json
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import flwr as fl
from collections import OrderedDict
import optuna
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# Partition 0: 634 images
# Partition 1: 529 images
# Partition 2: 887 images
# Partition 3: 732 images
# Partition 4: 670 images
# Partition 5: 584 images
# Partition 6: 717 images
# Partition 7: 667 images
# Partition 8: 588 images
# Partition 9: 592 images
# 5 splits - 4 for train and 1 for validation



partition_sizes = [
    634, 529, 887, 732, 670,
    584, 717, 667, 588, 592
]
train_sizes = [
    int(partition_sizes[i]*0.8) for i in range(len(partition_sizes))
]
val_sizes = [
    int(partition_sizes[i]*0.2) for i in range(len(partition_sizes))
]
num_clients = 1
num_rounds = 2
epochs_per_round = 1
# num_clients = 10
# num_rounds = 100
# epochs_per_round = 1

# lrs = [ 0.0001]*num_clients

print(nnUNet_raw, nnUNet_preprocessed, nnUNet_results)

# %% [code]
# TASK_ID = 1
PLANS_IDENTIFIER = "nnUNetPlans.json"
CONFIGURATION = "2d"
FOLD = 0
# lr=0.0001

# recently added to free up Memory
torch.cuda.empty_cache()

# # Set only GPUs 2, 4, and 7 as visible to PyTorch
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4,7"

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA Available")
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(num_devices):
        print(f"[{i}] {torch.cuda.get_device_name(i)}")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cuda")
print(f"Using device: {device}")


# %% [code]
def get_parameters(net) -> List[np.ndarray]:
    """Extracts model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    """Loads parameters into the model from a list of NumPy arrays."""
    state_dict = net.state_dict()  # Get existing state dict
    
    new_state_dict = OrderedDict(
        {k: torch.tensor(v, dtype=state_dict[k].dtype, device=state_dict[k].device)
         for k, v in zip(state_dict.keys(), parameters)}
    )
    
    net.load_state_dict(new_state_dict, strict=True)  # Enforce strict loading


# %% [code]
class FederatednnUNetTrainer(nnUNetTrainer):
    def __init__(self,plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
    def set_lr(self, lr: float):
        self.initial_lr= lr
    def load_model_from_state_dict(self, state_dict):
        """Load model from state dict, handling device mapping"""
        if isinstance(state_dict, OrderedDict):
            self.network.load_state_dict(state_dict)
        else:
            # Convert numpy arrays to tensors
            state_dict_tensors = OrderedDict(
                {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                 for k, v in state_dict.items()}
            )
            self.network.load_state_dict(state_dict_tensors)
        return self.network

    def run_only_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
        print(self.logger.my_fantastic_logging)
        metrics={
            "train_loss":float(self.logger.my_fantastic_logging['train_losses'][-1]),
            }
        self.on_train_end()
        return metrics
    def save_checkpoint(self, fname=None):
        """Save model checkpoint with client ID"""
        if fname is None:
            fname = join(self.output_folder, f"client_{self.client_id}_model_latest.model")
        return super().save_checkpoint(fname)
    

# %% [code]
class nnUNetFLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, lr: float = 0.0001):
        self.client_id = client_id+1
        self.lr = lr

        datadir = join("archive_v2/nnUNet_preprocessed", f"Dataset{str(self.client_id).zfill(3)}_Pneumonia")
        
        if not os.path.exists(datadir):
            shutil.copytree(join(nnUNet_preprocessed, f"Dataset{str(self.client_id).zfill(3)}_Pneumonia"), datadir)
        
        os.makedirs(datadir, exist_ok=True)
        
        self.trainer_kwargs = {
            "plans": join(datadir, PLANS_IDENTIFIER),
            "configuration": CONFIGURATION,
            "fold": FOLD,
            "dataset_json": json.load(open(join(datadir, "dataset.json"), "r")),
            "device": device,
        }

        self.trainer = FederatednnUNetTrainer(**self.trainer_kwargs)
        # self.trainer.initial_lr = lrs[self.client_id-1] if self.client_id!=-1 else 0.0001

        self.trainer.initialize()
        self.trainer.set_lr(self.lr)
        
        self.output_dir = join(nnUNet_results, f"client_{client_id}")
        os.makedirs(self.output_dir, exist_ok=True)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays"""
        state_dict = OrderedDict({
            k: torch.from_numpy(v)
            # k: torch.tensor(v, dtype=state_dict[k].dtype, device=state_dict[k].device)    
            for k, v in zip(self.trainer.network.state_dict().keys(), parameters)
        })
        # self.trainer.load_model_from_state_dict(state_dict)
        self.trainer.network.load_state_dict(state_dict)
        
        print(f"Client {self.client_id} received updated global model")
    
    def get_parameters(self, config) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays"""
        state_dict = self.trainer.network.state_dict()
        return [val.cpu().numpy() for _, val in state_dict.items()]
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the local dataset"""
        print(f"--> Client {self.client_id} starting local training...")
        
        self.set_parameters(parameters)
        
        epochs = int(config["epochs"]) if "epochs" in config else 1
        batch_size = int(config["batch_size"]) if "batch_size" in config else 12
        self.trainer.current_epoch=int(config["round_num"])
        self.trainer.num_epochs = self.trainer.current_epoch + epochs
        
        self.trainer.batch_size = batch_size
        metrics=self.trainer.run_only_training()
        
        self.trainer.save_checkpoint(join(self.output_dir, f"model_final.model"))
        
        updated_parameters = self.get_parameters(config)

        # lrs[self.client_id-1]=(-4.18*10**-5)*int(config["round_num"])+lrs[self.client_id-1]
        
        # on_train_end happens so dataset is empty 160 is hardcoded
        return updated_parameters, train_sizes[self.client_id-1], metrics                                   
    
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local validation dataset using nnUNetTrainer's validation process."""
        self.set_parameters(parameters)

        self.trainer.on_train_start()
        self.trainer.on_validation_epoch_start()
    
        val_outputs = []
        with torch.no_grad():
            for _ in range(self.trainer.num_val_iterations_per_epoch):
                batch = next(self.trainer.dataloader_val)  
                val_outputs.append(self.trainer.validation_step(batch)) 

        self.trainer.on_validation_epoch_end(val_outputs)
        print(self.trainer.logger.my_fantastic_logging)
        
        mean_fg_dice = float(self.trainer.logger.my_fantastic_logging['mean_fg_dice'][-1])
        loss_here =float(self.trainer.logger.my_fantastic_logging['val_losses'][-1])
        self.trainer.on_train_end()

        # on_train_end happens so dataset is empty 40 is hardcoded
        return loss_here-mean_fg_dice, val_sizes[self.client_id-1], {
            "dice_score": mean_fg_dice,
            "val_loss": loss_here
        }


# %%

def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict = {}) -> Tuple[float, int, Dict]:
    """
    Server-side evaluation using FederatednnUNetTrainer.
    This does NOT use nnUNetFLClient and is suitable for the Flower server.
    """

    print(f"Server-side evaluation at round {server_round}")

    # client=nnUNetFLClient(server_round)
    # return client.evaluate(parameters=parameters, config=config)

    dataset_id = server_round%10+1  # change with number of clients
    datadir = join("archive_v2/nnUNet_preprocessed", f"Dataset{str(dataset_id).zfill(3)}_Pneumonia")

    plans_path = join(datadir, PLANS_IDENTIFIER)
    dataset_json = json.load(open(join(datadir, "dataset.json"), "r"))
    # THIS DOESN'T WORK, SO I CHANGED THE LIBRARY CODE FOR THIS TO WORK, LOCATED AT: ~/miniconda3/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py
    # nnunetv2.paths.nnUNet_results="archive_v2/nnUNet_trained_models"  
    trainer = FederatednnUNetTrainer(
        plans=plans_path,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        device=device,
    )
    trainer.initialize()

    state_dict = OrderedDict({
        k: torch.from_numpy(v)
        for k, v in zip(trainer.network.state_dict().keys(), parameters)
    })
    trainer.network.load_state_dict(state_dict)

    trainer.on_train_start()
    trainer.on_validation_epoch_start()

    val_outputs = []
    with torch.no_grad():
        for _ in range(trainer.num_val_iterations_per_epoch):
            batch = next(trainer.dataloader_val)
            val_outputs.append(trainer.validation_step(batch))

    trainer.on_validation_epoch_end(val_outputs)
    trainer.on_train_end()

    mean_fg_dice = float(trainer.logger.my_fantastic_logging['mean_fg_dice'][-1])
    val_loss = float(trainer.logger.my_fantastic_logging['val_losses'][-1])
    os.makedirs(join(nnUNet_results, f"server_side"), exist_ok=True)
    if(float(os.environ["prev_dice"])<mean_fg_dice):
        trainer.save_checkpoint(join(nnUNet_results, f"server_side", f"model_final_{round(mean_fg_dice,4)}.model"))
        # print(f"New server side dice score: {mean_fg_dice}")
        print(f"\033[92mNew server side dice score: {mean_fg_dice}\033[0m")
        os.environ["prev_dice"]=str(mean_fg_dice)
    #logging
    metrics_path = join(nnUNet_results, "fl_history", "server_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            logs = json.load(f)
    else:
        logs = {"evaluation": {}, "training": {}}

    logs["evaluation"][f"round_{server_round}"] = {
        "num_clients": 1,
        "failures": 0,
        "results": [{
            "client": "server",
            "num_samples": 0,
            "metrics": {
                "val_loss": val_loss,
                "dice_score": mean_fg_dice
            }
        }]
    }

    with open(metrics_path, "w") as f:
        json.dump(logs, f, indent=4)

    return val_loss - mean_fg_dice, {
        "val_loss": val_loss,
        "dice_score": mean_fg_dice,
    }


# %%
class WeightedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_dir = os.path.join(nnUNet_results, "fl_history")
        os.makedirs(self.history_dir, exist_ok=True)
        # Initialize combined logs file
        self.combined_logs_path = os.path.join(self.history_dir, "metrics.json")
        self.combined_logs = {"evaluation": {}, "training": {}}

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics weighted by number of samples & save logs per round"""
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)

        evaluation_results = []
        for client_proxy, eval_res in results:
            if eval_res is not None:
                client_id = str(client_proxy)
                num_samples = eval_res.num_examples
                metrics = eval_res.metrics

                evaluation_results.append({
                    "client": client_id,
                    "num_samples": num_samples,
                    "metrics": {k: float(v) for k, v in metrics.items()}
                })

        # Update combined logs
        self.combined_logs["evaluation"][f"round_{server_round}"] = {
            "num_clients": len(results),
            "failures": len(failures),
            "evaluation_results": evaluation_results,
        }

        # Save combined logs
        with open(self.combined_logs_path, "w") as f:
            json.dump(self.combined_logs, f, indent=4)

        print(f"Updated combined evaluation logs with round {server_round} data")
        return aggregated_result

    def aggregate_fit(self, server_round, results, failures):
        aggregated_result = super().aggregate_fit(server_round, results, failures)

        fit_results = []
        for client_proxy, fit_res in results:
            if fit_res is not None:
                client_id = str(client_proxy)
                num_samples = fit_res.num_examples
                metrics = fit_res.metrics
                
                # Debug print
                print(f"Client {client_id} returned metrics: {metrics}")
                
                fit_results.append({
                    "client": client_id,
                    "num_samples": num_samples,
                    "metrics": {k: float(v) for k, v in metrics.items()}
                })

        # Update combined logs
        self.combined_logs["training"][f"round_{server_round}"] = {
            "train_results": fit_results
        }

        # Save combined logs
        with open(self.combined_logs_path, "w") as f:
            json.dump(self.combined_logs, f, indent=4)

        print(f"Updated combined training logs with round {server_round} data")
        return aggregated_result
# %%

def start_federated_learning(num_clients: int = 1, num_rounds: int = 2, epochs_per_round: int = 1):
    """Start the federated learning process"""
    print(f"Starting Federated Learning with {num_clients} clients for {num_rounds} rounds...")    
    
    def client_fn(cid: str) -> fl.client.Client:
        client_id = int(cid)
        return nnUNetFLClient(
            client_id=client_id,
        )

    # strategy = WeightedFedAvg(
    #     fraction_fit=0.5,  
    #     fraction_evaluate=0.3,  
    #     min_fit_clients=num_clients*0.3,  
    #     min_evaluate_clients=num_clients*0.2,  
    #     min_available_clients=num_clients,  
    #     on_fit_config_fn=lambda round_num: {
    #         "epochs": epochs_per_round,
    #         "batch_size": 12,
    #         "round_num": round_num
    #     },
    #     on_evaluate_config_fn=lambda round_num: {
    #         "round_num": round_num
    #     },
    #     evaluate_fn=evaluate,
    # )
    # TESTING
    strategy = WeightedFedAvg(
        fraction_fit=1,  
        fraction_evaluate=1,  
        min_fit_clients=num_clients*1,  
        min_evaluate_clients=num_clients*1,  
        min_available_clients=num_clients,  
        on_fit_config_fn=lambda round_num: {
            "epochs": epochs_per_round,
            "batch_size": 12,
            "round_num": round_num
        },
        on_evaluate_config_fn=lambda round_num: {
            "round_num": round_num
        },
        evaluate_fn=evaluate,
    )
    # Start simulation
    history =fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        # Resources present in DGX V100 are 8 GPUS
        # Number of CUDA devices: 8
        # [0] Tesla V100-SXM2-32GB
        # [1] Tesla V100-SXM2-32GB
        # [2] Tesla V100-SXM2-32GB
        # [3] Tesla V100-SXM2-32GB
        # [4] Tesla V100-SXM2-32GB
        # [5] Tesla V100-SXM2-32GB
        # [6] Tesla V100-SXM2-32GB
        # [7] Tesla V100-SXM2-32GB
        client_resources={"num_cpus": 5, "num_gpus": 3},  
    )


    history_dir = join(nnUNet_results, "fl_history")
    os.makedirs(history_dir, exist_ok=True)
    print("Federated Learning completed successfully!")
    return history
# %%

def show_predictions():
    """Visualize predictions for a specific client"""
    client_id = 11  # Untouched data
    datadir = join("archive_v2/nnUNet_preprocessed", f"Dataset{str(client_id).zfill(3)}_Pneumonia")
    
    # Load the model
    trainer = FederatednnUNetTrainer(
        plans=join(datadir, PLANS_IDENTIFIER),
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=json.load(open(join(datadir, "dataset.json"), "r")),
        device=device,
    )
    trainer.initialize()
    
    # Load the model with highest dice score
    models = sorted(glob.glob(join(nnUNet_results, f"server_side", "model_final_*.model")), key=os.path.getmtime)
    model_path = models[-1] if models else None
    if model_path is None:
        print("No model found for visualization.")
        return
    print(f"Loading model from {model_path}")
    # trainer.load_model_from_state_dict(torch.load(model_path))
    # parameters = torch.load(model_path)
    # state_dict = OrderedDict({
    #     k: torch.from_numpy(v)
    #     for k, v in zip(trainer.network.state_dict().keys(), parameters)
    # })
    # trainer.network.load_state_dict(state_dict)
    trainer.load_checkpoint(model_path)
    # Visualize predictions (dummy code, replace with actual visualization)
    print(f"Model loaded for client {client_id}. Ready to visualize predictions.")

    trainer.on_train_start()
    trainer.on_validation_epoch_start()
    n=30

    val_outputs = []
    y=[]
    with torch.no_grad():
        for _ in range(trainer.num_val_iterations_per_epoch):
            batch = next(trainer.dataloader_val)
            x=trainer.validation_step_post_training(batch)
            # detach the tensors
            x['mask'] = x['mask'].detach().cpu()
            x['data'] = x['data'].detach().cpu()
            for i in range(len(x['target'])):
                for j in range(len(x['target'][i])):
                    x['target'][i][j] = x['target'][i][j].detach().cpu()
            y.append(x)
            if len(val_outputs) <= n:
                val_outputs.append(x)
    for i in range(n):
        pred = val_outputs[i]['mask']
        image = val_outputs[i]['data']
        mask = val_outputs[i]['target']
        print(f"[DEBUG] i={i} | image shape: {image.shape}, pred shape: {pred.shape}")
        for j in range(3):
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow((image[j][0]), cmap='gray')
            plt.title("Original Image")

            plt.subplot(1, 3, 2)
            plt.imshow((mask[0][j][0])*255, cmap='gray')
            plt.title("Ground Truth Mask")

            plt.subplot(1, 3, 3)
            plt.imshow((pred[j][0]-2)*-255, cmap='gray')
            plt.title("Predicted Mask")

            plt.tight_layout()
            plt.savefig(join(nnUNet_results, "server_side", f"predictions_{i}.png"))
            plt.close()

    trainer.on_validation_epoch_end(y)
    trainer.on_train_end()

    mean_fg_dice = float(trainer.logger.my_fantastic_logging['mean_fg_dice'][-1])
    val_loss = float(trainer.logger.my_fantastic_logging['val_losses'][-1])

    print(f"Mean FG Dice: {mean_fg_dice}")
    print(f"Validation Loss: {val_loss}")



# %% [code] {"jupyter":{"outputs_hidden":false}}
if __name__ == "__main__":
            # TESTING
    history = start_federated_learning(
        num_clients=num_clients,
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round
    )
    show_predictions()
