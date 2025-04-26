# Skeleton Transformer Anomaly Detector

## Setup

> [!NOTE]
> If running in Google Colab, install PyTorch Lightning first:
> ```sh
> pip install pytorch-lightning
> ```

### Datasets
You can download the extracted poses for the datasets HR-Avenue and HR-ShanghaiTech from the [GDRive](https://drive.google.com/drive/folders/1aUDiyi2FCc6nKTNuhMvpGG_zLZzMMc83?usp=drive_link).

Place the extracted folder in a `./data` folder and change the configs accordingly.

### **Training** 

In each config file you can set the hyperparameters. The default parameters achieve the best results reported in the paper.

Update the args 'data_dir', 'test_path', 'dataset_path_to_robust' with the path where you stored the datasets.

To train STAD:
```sh
python train.py --config config/[Avenue/STC]/{config_name}.yaml
```


### Once trained, you can run the **Evaluation**

The training config is saved the associated experiment directory (`/args.exp_dir/args.dataset_choice/args.dir_name`). 
To evaluate the model on the test set, you need to change the following parameters in the config:

- split: 'Test'
- validation: 'False'
- load_ckpt: 'name_of_ckpt'

Test STAD
```sh
python eval.py --config /args.exp_dir/args.dataset_choice/args.dir_name/config.yaml
```
