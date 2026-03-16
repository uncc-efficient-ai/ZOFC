## ZOFC
This is implmentation of our WACV2026 paper: <a href="https://www.arxiv.org/abs/2510.21019">[More Than Memory Savings: Zeroth-Order Optimization Mitigates Forgetting in Continual Learning]</a>.

## how to use

### Dependencies

1. [torch 2.0.1](https://github.com/pytorch/pytorch)
2. [torchvision 0.15.2](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)
4. [tqdm](https://github.com/tqdm/tqdm)
5. [numpy 1.23.5](https://github.com/numpy/numpy)
6. [scipy](https://github.com/scipy/scipy)
7. [easydict](https://github.com/makinacorpus/easydict)

Also, see requirements.txt

### Run experiment

1. Edit the `[MODEL NAME].json` file for global settings and hyperparameters.
2. Run:

    ```bash
    python main.py --config=./exps/[MODEL NAME].json
    ```

For ZOFC reproductions, you can first run  
1. python main.py --config=./exps/newlae_hybrid_zafc_cifar_5.json
2. python main.py --config=./exps/newlae_hybrid_zafc_cifar_10.json
3. python main.py --config=./exps/newlae_hybrid_zafc_inr_5.json
4. python main.py --config=./exps/newlae_hybrid_zafc_inr_10.json

### Datasets
We use Cifar100, ImageNet-R, DomainNet.
- **CIFAR100**: will be automatically downloaded by the code.
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **DomainNet**: download from [link](http://ai.bu.edu/M3SDA/), place it into data/ folder

> These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.
>
> When training **not** on `CIFAR100`, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```

### Acknowledgement
We would like to thank the following repos providing helpful components/functions in our work:
- <a href="https://github.com/LAMDA-CL/LAMDA-PILOT.git">[PILOT]</a>.
- <a href="https://github.com/liangyanshuo/InfLoRA.git">[InfLoRA]</a>.
