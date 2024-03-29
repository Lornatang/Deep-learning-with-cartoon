# Deep-learning-with-cartoon

This is the repository for a project that I am currently working on to verify whether the indicators in the current paper meet my expectations. 
I used the current mainstream four GAN models (DCGAN, WGAN, wgan-gp and LSGAN) to process the cat data I collected.
All current models have been made into visual versions and are implemented using Django.

### Objectives

- Generate images of cats for Generative Adversarial Networks([GAN](https://arvix.org/pdf/1511.06434.pdf))

    - [DCGAN](https://github.com/lornatang/Deep-learning-with-cartoon/dcgan.py)
    - [WGAN](https://github.com/lornatang/Deep-learning-with-cartoon/wgan.py)
    - [WGAN-GP](https://github.com/lornatang/Deep-learning-with-cartoon/wgan_gp.py)
    - [LSGAN](https://github.com/lornatang/Deep-learning-with-cartoon/lsgan.py)
    
- Requirement

    - **Python >= 3.7**
    - Pytorch >= 1.3
    - Torchvision >= 0.4
    - [Face dataset](https://pan.baidu.com/s/1ej0tYJWs4L8pE199s8kE7A) password:`llot`
    
- Run
```text
git clone https://github.com/lornatang/Deep-learning-with-cartoon.git
cd Deep-learning-with-cats/
# train dcgan 64 x 64 model.
python3 dcgan.py  --cuda 

# train wgan 64 x 64 model.
python3 wgan.py --cuda 

# train wgan_gp 64 x 64 model.
python3 wgan_gp.py --cuda 
```

