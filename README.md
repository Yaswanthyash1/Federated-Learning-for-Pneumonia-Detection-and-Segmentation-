# Federated-Learning-for-Pneumonia-Detection-and-Segmentation


- Install dependencies
    ```bash
    python3 -m venv flower
    sudo apt install  python3.10-venv
    python3 -m pip install ipykernel
    pip install pillow numpy matplotlib typing nnunetv2 flwr flwr[simulation] torch torchvision ipykernel

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

- Download the dataset
    ```bash
    kaggle datasets download -d nnagabhushanam/nnunet-lung-segmentation-dataset-rsna
    unzip nnunet-lung-segmentation-dataset-rsna.zip 
    ```
    this will extract to the dir dataset_nnunet in the current directory.

- Use the py script `dataset_converter.py` to convert the dataset to nnUNet format

- Export the following variables:
    ```bash
    export nnUNet_raw="/archive_v4/nnUNet_raw_data_base"
    export nnUNet_preprocessed="/archive_v4/nnUNet_preprocessed"
    export nnUNet_results="/archive_v4/nnUNet_trained_models"
    ```