import json
import matplotlib.pyplot as plt
import sys

client_file = sys.argv[1]
server_file = sys.argv[2]

with open(client_file, "r") as f:
    client_data = json.load(f)

rounds = []
avg_dice_scores = []
avg_val_losses = []

for rnd, details in client_data["evaluation"].items():
    total_samples = 0
    total_dice = 0
    total_loss = 0

    for result in details["evaluation_results"]:
        samples = result["num_samples"]
        dice = result["metrics"]["dice_score"]
        val_loss = result["metrics"]["val_loss"]
        total_samples += samples
        total_dice += dice * samples
        total_loss += val_loss * samples

    if total_samples > 0:
        rounds.append(int(rnd.split("_")[1]))
        avg_dice_scores.append(total_dice / total_samples)
        avg_val_losses.append(total_loss / total_samples)

sorted_data = sorted(zip(rounds, avg_dice_scores, avg_val_losses))
rounds, avg_dice_scores, avg_val_losses = zip(*sorted_data)

rounds2 = []
avg_train_losses = []

for rnd, details in client_data["training"].items():
    total_samples = 0
    total_loss = 0

    for result in details["train_results"]:
        samples = result["num_samples"]
        val_loss = result["metrics"]["train_loss"]
        total_samples += samples
        total_loss += val_loss * samples

    if total_samples > 0:
        rounds2.append(int(rnd.split("_")[1]))
        avg_train_losses.append(total_loss / total_samples)

sorted_data = sorted(zip(rounds2, avg_train_losses))
rounds2, avg_train_losses = zip(*sorted_data)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rounds, avg_dice_scores, label='Avg Dice Score')
plt.xlabel("Round")
plt.ylabel("Dice Score")
plt.title("Client Weighted Avg Dice Score per Round")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rounds, avg_val_losses, color='orange', label='Avg Validation Loss')
plt.plot(rounds2, avg_train_losses, color='blue', label='Avg Training Loss')
plt.xlabel("Round")
plt.ylabel("Loss")
plt.title("Client Weighted Avg Loss per Round")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("client_metrics.png")

with open(server_file, "r") as f:
    server_data = json.load(f)

rounds = []
val_losses = []
dice_scores = []

for rnd, details in server_data["evaluation"].items():
    round_num = int(rnd.split("_")[1])
    result = details["results"][0]["metrics"]
    val_losses.append(result["val_loss"])
    dice_scores.append(result["dice_score"])
    rounds.append(round_num)

sorted_data = sorted(zip(rounds, val_losses, dice_scores))
rounds, val_losses, dice_scores = zip(*sorted_data)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rounds, dice_scores, label='Dice Score', color='green')
plt.xlabel("Round")
plt.ylabel("Dice Score")
plt.title("Server Dice Score per Round")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(rounds, val_losses, label='Validation Loss', color='purple')
plt.xlabel("Round")
plt.ylabel("Validation Loss")
plt.title("Server Validation Loss per Round")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("server_metrics.png")

