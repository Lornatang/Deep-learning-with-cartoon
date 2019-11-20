# Deep-learning-with-cartoon

This is the repository for a project that I am currently working on to verify whether the indicators in the current paper meet my expectations. 
I used the current mainstream four GAN models (DCGAN, WGAN, wgan-gp and LSGAN) to process the cat data I collected.
All current models have been made into visual versions and are implemented using Django.

### Objectives

- Generate images of cats for Generative Adversarial Networks([GAN](https://arvix.org/pdf/1511.06434.pdf))

    - [DCGAN](https://github.com/lornatang/Deep-learning-with-cats/dcgan.py)
    - [WGAN](https://github.com/lornatang/Deep-learning-with-cats/wgan.py)
    - [WGAN-GP](https://github.com/lornatang/Deep-learning-with-cats/wgan_gp.py)
    - [LSGAN](https://github.com/lornatang/Deep-learning-with-cats/lsgan.py)
    
- Requirement

    - **Python >= 3.7**
    - Pytorch >= 1.3
    - Torchvision >= 0.4
    - [Face dataset](https://www.kaggle.com/tongpython/cat-and-dog/download)
    
- Run
```text
git clone https://github.com/lornatang/Deep-learning-with-cats.git
cd Deep-learning-with-cats/
# train dcgan 64 x 64 model.
python3 dcgan.py --img_size 64 --cuda 
# train dcgan 128 x 128 model.
python3 dcgan.py --img_size 128 --cuda 

# train wgan 64 x 64 model.
python3 wgan.py --img_size 64 --cuda 
# train wgan 128 x 128 model.
python3 wgan.py --img_size 128 --cuda 

# train wgan_gp 64 x 64 model.
python3 wgan_gp.py --img_size 64 --cuda 
# train wgan_gp 128 x 128 model.
python3 wgan_gp.py --img_size 128 --cuda 
```

