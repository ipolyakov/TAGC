# THC
## Prerequisites:
* Pytorch 2.4.1 , install with pip
## Apply path to Pytorch
```
pip show torch
```
Get the path to torch installation such as ~/.local/lib/python3.12/site-packages from the pip show output.
Then use the path to apply the patch:
```
git apply --unsafe-paths --directory ~/.local/lib/python3.12/site-packages thc/patches/torch.diff
```

Install HLC and THC:
```
cd thc
git submodule update --init
pip install lossless_homomorphic_compression/lossless_homomorphic_compression_api
pip install .
python nanoGPT/data/openwebtext/prepare.py
```

Run training within a two ranks setup:
```
torchrun --rdzv-backend=c10d --rdzv-endpoint=<IP-of-the-master-rank> --nnodes 2 --nproc-per-node=1 train.py config/train_gpt2.py
```
