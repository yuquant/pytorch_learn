#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.nvidia.com/dli"> <img src="DLI Header.png" alt="Header" style="width: 400px;"/> </a>

# # Data Augmentation and Segmentation with Generative Networks for Medical Imaging
# ## Outline
# This lab demonstrates two use cases for Generative Adversarial Networks (GANs) in medical imaging:
# <ol>
#     <li> Generating randomized brain MRI images from random noise using a GAN.</li>
#     <li> Translating from one image domain to another with a conditional GAN (pix2pix).</li>
#     This technique is applied to various tasks including
#     <ol>
#         <li> Segmenting brain anatomy (white matter, gray matter, CSF). </li>
#         <li> Generating brain MRI from the segmentation. </li>
#         <li> Augmenting the translation of image modalities in a limited dataset to perform ischemic stroke segmentation. </li>
#     </ol>
# </ol>
# 

# ## Background
# Generative Adversarial Networks (GAN) were first introduced by Ian Goodfellow et al, in 2014 (https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf).
# 
# It was shown that random handwritten digits could be generated from the generator network of a GAN, after training on the MNIST dataset (http://yann.lecun.com/exdb/mnist/).
# 
# ### Preparation
# We first need to load some libraries and perform a few final data preparation steps. You can read on in the notebook while these steps are running.

# In[1]:


import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import SimpleITK as sitk
from IPython import display
from IPython.display import Image
get_ipython().system('./datasetup.sh')


# ## 1. Generating random T1-weighted brain MRI
# 
# We will first train a GAN to generate random 2-dimensional T1-weighted brain MRIs.
# 
# The T1-weighted brain MRIs will be generated from random noise, shown as "z" in the picture below.
# 
# <img src="GAN1.png">
# 
# We train the Generator and Discriminator concurrently.
# 
# The Discriminator is trained to distinguish "real" and "generated" brain MRIs, and the Generator is trained to win over the Discrimiator, i.e., generate more realistic brain MRI so that the Discriminator cannot distinguish them from real brain MRIs.
# 
# The code is here: <a href = "gan2d.py">`gan2d.py`</a>.
# 
# We start from 145th epoch, to save time - the network will learn to generate realistic brain MRI images after about 150 epochs, and currently training one epoch takes about a minute.<br/>
# You can let it train for about 5 minutes, or you can also <u style="color:red">Stop training</u> (by using the stop button 	&#11035; in the toolbar above). We will need to interrupt training a few times in the rest of the lesson, so please make note.

# In[2]:


get_ipython().system('python3 gan2d.py --data_dir=/dli/data/png/t1 --restore_checkpoints=True --start_epoch=145')


# Let's see how the images get more realistic as the epochs progress.

# In[3]:


with imageio.get_writer('dcgan.gif', mode='I') as writer:
    filenames = glob.glob('/dli/data/gan2d/gan2d_images/image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = i
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
    
# this is a hack to display the gif inside the notebook
last = os.system('cp dcgan.gif dcgan.gif.png')
# display
Image(filename="dcgan.gif.png")


# ## 2. Generate brain segmentation from T1-weighted MRI.
# 
# We used a GAN to generate random brain MRIs. However, we may want to have more control over what we generate.
# 
# A conditional GAN generates an output based on a given input.
# 
# For instance, "pix2pix" (https://arxiv.org/abs/1611.07004) conditional GAN gets an image as input and generates a new image as output.
# 
# In the original "pix2pix" paper, the authors demonstrated translating street scenes to labels and vice versa, or black-and-white images to color images, for example.
# 
# We will now segment T1-weighted brain MRI using the pix2pix conditional GAN.
# 
# Training the conditional GAN is mostly similar to training the original GAN, except we give an image input to the generator, whereas in the original GAN images were generated from a noise vector.
# 
# <img src="GAN2.png">
# 
# As above, we should <u style="color:red">stop training</u> after about two epochs (one epoch takes about 1.5 mins) due to time constraints. (NOTE: In subsequent exercises, the final image produced will be labeled with a number one lower than its corresponding epoch. If you stop training after epoch 5, for example, the final image will be saved as `image_at_epoch_0004.png`. Adjust `Image` commands appropriately.)

# In[8]:


get_ipython().system('python3 pix2pix2d.py --data_dir_A=/dli/data/nifti-png3ch/t1/ --data_dir_B=/dli/data/nifti-png3ch/seg/ --use_partial_data=True')


# Let's see how the segmentation looks like - it'll learn to generate images much closer to the ground truth when we train for more epochs. You can change the final digit in the filename (default is 0) to 0-9 to see different examples.

# In[9]:


Image(filename='image_at_epoch_0001-0.png') 


# ## 3. Generating T1-weighted brain MRI from segmentation label.
# 
# The conditional GAN framework is quite powerful, in that input and output can be anything if they are related.
# 
# Instead of generating brain segmentation from T1-weighted image (like in a normal image segmentation network), this time we will generate T1-weighted brain MRI from a segmentation.
# 
# We can do this by simply reversing the input and output to generate brain MRIs from labels.
# 
# <img src="GAN3.png">
# 
# Again, <u style="color:red">Stop training</u> after 2 epochs (takes about 3 minutes) due to time limits.

# In[10]:


get_ipython().system('python3 pix2pix2d.py --data_dir_A=/dli/data/nifti-png3ch/seg/ --data_dir_B=/dli/data/nifti-png3ch/t1/ --use_partial_data=True')


# Let's see how the generated images look. <br/>
# After training about 2 epochs, it already learned to generate the brain from brain segmentation, but the areas other than the brain are blurry.

# In[11]:


Image(filename='image_at_epoch_0001-0.png') 


# ## 4. Ischemic stroke segmentation from limited data
# 
# From http://www.isles-challenge.org/:
# 
# <blockquote>Ischemic Stroke Lesion Segmentation (ISLES) challenge (http://www.isles-challenge.org/) is a 
# medical image segmentation challenge at the International Conference on Medical Image Computing 
# and Computer Assisted Intervention (MICCAI) 2018
# This year ISLES 2018 asks for methods that allow the segmentation of stroke lesions based on 
# acute CT perfusion data. Therefore, a new data set of 103 stroke patients and matching expert 
# segmentations are provided.
# 
# Training data set consists of 63 patients. Some patient cases have two slabs to cover the stroke
# lesion. These are non-, or partially-overlapping brain regions. Slabs per patient are indicated
# with letters "A" and "B" for first and second slab, respectively. The mapping between case number 
# and training name is also provided at SMIR (e.g. Train_40_A = case 64; Train_40_B = case 65).
# Developed techniques will be evaluated by means of a testing set including 40 stroke cases.
# Acquired modalities are described in detail below.
# 
# GOLD STANDARD: DIFFUSION MAPS (DWI)
# Infarcted brain tissue can be recognised as hyperintense regions of the DWI trace images (DWI maps). 
# Provided ground-truth segmentation maps were manually drawn on those scans.
# 
# PERFUSION MAPS (CBF, MTT, CBV, TMAX, CTP SOURCE DATA)
# To assess cerebral perfusion, a contrast agent (CA) is administered to the patient and its temporal
# change is captured in dynamic scans acquired 1-2 sec apart. Subsequently, perfusion maps are derived
# from these raw data for clinical interpretation. Different maps aim to yield different information, 
# and the most commonly calculated maps include cerebral blood volume (CBV), cerebral blood flow (CBF), 
# and time to peak of the residue function (Tmax). These perfusion maps serve as input to the algorithms.
# </blockquote>

# It is a challenging task - input data is high-dimensional, and the dimension varies patient by patient.
# Let's first have a look into the dataset.

# In[12]:


get_ipython().system('ls /dli/data/ISLES2018/TRAINING')


# In[13]:


get_ipython().system('ls /dli/data/ISLES2018/TESTING')


# Training dataset has 94 cases, and testing dataset has 62 cases.<br/>
# Let's see how each case looks like.

# In[14]:


get_ipython().system('ls /dli/data/ISLES2018/TRAINING/case_1')


# In[15]:


get_ipython().system('ls /dli/data/ISLES2018/TESTING/case_1')


# In Training dataset we have CT (SMIR.Brain.XX.O.CT.339203), MR Perfusion (SMIR.Brain.XX.O.MR_4DPWI.339202), and other perfusion maps derived from that (CBF: cerebral blood flow; CBV: cerebral blood volume; MTT: mean transit time; Tmax: time to peak of the residue function). We also have ground-truth stroke segmentation label (SMIR.Brain.XX.O.OT.339208).
# 
# In Testing dataset we have CT (SMIR.Brain.XX.O.CT.346291), then CT Perfusion instead of MR (SMIR.Brain.XX.O.CT_4DPWI.346290), and same sort of perfusion maps derived from that (CBF, CBV, MTT, Tmax).

# The background of this setting is (correct me if I'm wrong if you're a neuroradiologist) - <br/>
# CT image is easier and faster to obtain than MRI. <br/>
# Stroke, specifically dead tissue, needs to be diagnosed fast in order to decide how to treat the patient, and so CT is used in the operating room over MRI. <br/>
# However, stroke is best seen on MRI, so the gold standard is diffusion weighted MRI.
# 
# From http://www.isles-challenge.org/:
# 
# <blockquote>Infarcted brain tissue can be recognised as hyperintense regions of the DWI trace images (DWI maps). Provided ground-truth segmentation maps were manually drawn on those scans.
# </blockquote>
# 
# The main goal of this challenge is to segment stroke from CT source images, which will have real clinical benefit.

# Now, let's see the dimensions of each image.

# In[16]:


def read_nii_from_file(filename, is_label=False):
    sitk_niim = sitk.ReadImage(filename)
    niim = sitk.GetArrayFromImage(sitk_niim)
    return niim


# In[17]:


for diri in glob.glob('/dli/data/ISLES2018/TRAINING/case_1/*'):
    imfname = diri.split('/')[-1] + '.nii'
    imfulldir = os.path.join(diri, imfname)
    im = read_nii_from_file(imfulldir)
    print(imfname, im.shape)


# In[18]:


for diri in glob.glob('/dli/data/ISLES2018/TESTING/case_1/*'):
    imfname = diri.split('/')[-1] + '.nii'
    imfulldir = os.path.join(diri, imfname)
    im = read_nii_from_file(imfulldir)
    print(imfname, im.shape)


# We can see that the (x, y) dimension of the 3D images are consistant as 256x256, but the z dimension varies from case to case.<br/>
# Also, the perfusion image's 4th dimension varies from case to case.
# 
# Now the task is to generate stroke segmentation from this high-dimensional data.
# 
# Let's see how the images look like:

# In[19]:


plt.imshow(read_nii_from_file('/dli/data/ISLES2018/TRAINING/case_1/SMIR.Brain.XX.O.CT.339203/SMIR.Brain.XX.O.CT.339203.nii')[4,:,:], cmap='gray')


# In[20]:


plt.imshow(read_nii_from_file('/dli/data/ISLES2018/TRAINING/case_1/SMIR.Brain.XX.O.MR_MTT.339207/SMIR.Brain.XX.O.MR_MTT.339207.nii')[4,:,:], cmap='gray')


# In[21]:


plt.imshow(read_nii_from_file('/dli/data/ISLES2018/TRAINING/case_1/SMIR.Brain.XX.O.MR_Tmax.339209/SMIR.Brain.XX.O.MR_Tmax.339209.nii')[4,:,:], cmap='gray')


# In[22]:


plt.imshow(read_nii_from_file('/dli/data/ISLES2018/TRAINING/case_1/SMIR.Brain.XX.O.OT.339208/SMIR.Brain.XX.O.OT.339208.nii')[4,:,:], cmap='gray')


# Since the data is high-dimensional (3D + multiple-modalities = 4D), and the stroke lesion we want to
# segment is so small, the standard segmentation algorithms don't do well.
# 
# To see this, let's apply the pix2pix out of the box, which we used to segment brain anatomy, to the new task of segmenting stroke lesions.

# For easier data processing and visualization, we convert the 4-dimensional perfusion image to a 3-dimensional image, by Principal Component Analysis (PCA).
# 
# From Wikipedia (https://en.wikipedia.org/wiki/Dimensionality_reduction)
# <blockquote>Principal component analysis (PCA):<br/>
# The main linear technique for dimensionality reduction, principal component analysis, performs a linear mapping of the data to a lower-dimensional space in such a way that the variance of the data in the low-dimensional representation is maximized. In practice, the covariance (and sometimes the correlation) matrix of the data is constructed and the eigenvectors on this matrix are computed. The eigenvectors that correspond to the largest eigenvalues (the principal components) can now be used to reconstruct a large fraction of the variance of the original data. Moreover, the first few eigenvectors can often be interpreted in terms of the large-scale physical behavior of the system. The original space (with dimension of the number of points) has been reduced (with data loss, but hopefully retaining the most important variance) to the space spanned by a few eigenvectors.
# </blockquote>

# First we sample 2D images from the 4D ISLES data.<br/>
# We select CT, Perfusion image, Tmax and convert them to RGB channels, where the dimension of the perfusion image is redueced from 4D to 3D using PCA.<br/>
# The code for sampling the images is here: <a href = "utils/isles18_sample_2d.ipynb">utils/isles18_sample_2d.ipynb</a>.
# 
# Let's <u style="color:red">stop training</u> after about 15 epochs (~ 10 min).

# In[25]:


get_ipython().system('python3 pix2pix2d.py --data_dir_A=/dli/data/training_png/img --data_dir_B=/dli/data/training_png/seg --output_file_dir=isles_imgtoseg1 --save_slides_with_lesion_only=True')


# Now let's see some results.

# In[26]:


Image(filename='isles_imgtoseg1/image_at_epoch_0014-0.png') 


# In[27]:


Image(filename='isles_imgtoseg1/image_at_epoch_0014-5.png') 


# The performance isn't ideal; in fact, it's not segmenting anything!<br/>
# One of the reasons is that in pix2pix algorithm discriminator is trained patch-wise (70x70 size patches in 2D case), and so the loss becomes too unbalanced and small with this task - there is only a quite small stroke region while most of the image is background (non-stroke region).
# 
# In order to mitigate this problem we first translate from one image modality to another, and transfer learn from that to segment strokes.
# 
# We sample image-modality-1: [CT/Perfusion/Tmax] to image-modality-2: [CBF/CBV/MTT], then translate [CT/Perfusion/Tmax] to [Perfusion-first-image/stroke-label/Perfusion-last-image].<br/>
# Perfusion images are 4D: their sizes are 256x256x20x40
# With Perfusion-first, we take the first image of Perfusion in its 4th dimension and with Perfusion-last, the last image in its 4th dimension. 
# 
# We do [Perfusion-first-image/stroke-label/Perfusion-last-images] so the target is more reasonably balanced and loss doesn't become too small. But we also randomly switch the 1st and 3rd channel of the target domain (perf-1st, perf-last), to introduce noise - making sure they are not our main target to optimize for - our main target is the stroke region.
# 
# <img src="GAN4.png">
# 
# Now let's train what translating [CT/Perfusion/Tmax] to [CBF/CBV/MTT] looks like, which we will later use to transfer-learn to translate [CT/Perfusion/Tmax] to [Perfusion-first-image/stroke-label/Perfusion-last-image] for our final stroke segmentation.
# 
# Let's <u style="color:red">stop training</u> after about 15 epochs (~10 min).

# In[28]:


get_ipython().system('python3 pix2pix2d.py --data_dir_A=/dli/data/training_png/img --data_dir_B=/dli/data/training_png/img2 --output_file_dir=isles_imgtoimg2 --save_slides_with_lesion_only=True')


# Let's see the images.

# In[29]:


Image(filename='isles_imgtoimg2/image_at_epoch_0014-0.png')


# In[30]:


Image(filename='isles_imgtoimg2/image_at_epoch_0014-5.png')


# To give the transfer learning exercises below the best possible pretrained models to work with, we already trained several models for 200 epochs. To prepare the files for use, run the command below once.

# In[31]:


get_ipython().system('./checkpointsetup.sh')


# Let's compare the advantages of a pretrained model on segmentation from image-modality-1 to [Perfusion-first/Label/Perfusion-last]. We use the flag,
# 
# `--swap_noise_imB_channel_13=True` so we randomly swap first and last channel (except on the stroke label) to add noise to those channels so the network can learn to focus on segmenting the stroke label. Try it without this option, too.
# 
# In the second command, we use
# 
# `--restore_checkpoints=True` to enable transfer learning (or loading of a pretrained model).
# 
# Run each of these commands for 2 epochs, then <u style="color:red">stop training</u>.

# In[32]:


get_ipython().system('python3 pix2pix2d.py --data_dir_A=/dli/data/training_png/img --data_dir_B=/dli/data/training_png/seg2 --output_file_dir=isles_img2seg2 --swap_noise_imB_channel_13=True --save_slides_with_lesion_only=True')


# In[33]:


Image(filename='isles_img2seg2/image_at_epoch_0001-0.png') 


# In[34]:


Image(filename='isles_img2seg2/image_at_epoch_0001-5.png') 


# To implement transfer learning from the pretrained models, we need to use the flag `--restore_checkpoints=True`. In this case, the checkpoints will be loaded from the same directory that we flag as `--output_file_dir`.

# In[35]:


get_ipython().system('python3 pix2pix2d.py --data_dir_A=/dli/data/training_png/img --data_dir_B=/dli/data/training_png/seg2 --output_file_dir=/dli/data/img2label_2d --restore_checkpoints=True --swap_noise_imB_channel_13=True --save_slides_with_lesion_only=True')


# In[36]:


Image(filename='/dli/data/img2label_2d/image_at_epoch_0001-1.png')


# In[37]:


Image(filename='/dli/data/img2label_2d/image_at_epoch_0001-9.png')


# The result will look better if we train on 3D, which will be left as an exercise.<br/>
# All the code is already there; please see below.
# 
# Before we continue, let's now see how it applies to test cases when we have CT perfusion images instead of MR perfusion images.

# In[38]:


get_ipython().system('./checkpointsetup_test.sh')


# In[39]:


get_ipython().system('python3 pix2pix2d.py --data_dir_A=/dli/data/testing_png/img --data_dir_B=/dli/data/testing_png/seg2 --test_data_dir_A=/dli/data/testing_png/img --test_data_dir_B=/dli/data/testing_png/seg2 --output_file_dir=/dli/data/img2label_2d_test --restore_checkpoints=True --train_or_test=test')


# In[40]:


Image(filename='/dli/data/img2label_2d_test/image_at_epoch_case_19_3ch_sli0.png')


# In[41]:


Image(filename='/dli/data/img2label_2d_test/image_at_epoch_case_25_3ch_sli1.png')


# See that it successfully segments stroke from CT perfusion data. <br/>
# Please note that the "Ground Truth" does not show the stroke region, since it's the test dataset - not showing the answer to the challenge competitors.

# ## Other exercises to try
# We have included several other types of pretrained models that can be used to further experiment with the abilities of GANs. These can be used as pretrained models, or implemented for transfer learning. For example, we can get better training and prediction by giving the generator the additional shape context from the 3D scans.
# 
# Pretrained models are loaded by setting `--restore_checkpoints=True` and `--output_file_dir=`
# 
# `/dli/data/img2img2_2d` image modality 1 -> 2 in 2d, use with `pix2pix2d.py`
# 
# `/dli/data/img2label_2d` image -> segmentation in 2d, use with `pix2pix2d.py`
# 
# `/dli/data/img2img2_4d` image modality 1 -> 2 in 3d, use with `pix2pix3d.py`
# 
# `/dli/data/img2seg_4d` image -> segmentation in 3d, use with `pix2pix3d.py`
# 
# Be sure to match these models with data of the appropriate type, found in the flags `data_dir_A` and `_B` in the cells above.

# The pix2pix by default uses U-Net (https://arxiv.org/abs/1505.04597) architecture for its generator.
# We can change the generator to use ResNet (https://arxiv.org/abs/1512.03385) architecture with the following flag,
# 
# `--generator_type=resnet`
# 
# We have translated the images from one modality to another, and attached DWI images to the stroke segmentation label to make loss more reasonable while balancing the target.
# 
# Can we try sampling an equal number of patches on stroke vs. not-stroke to make them balanced?
# 
# `--sample_balanced=True`
# 
# This method takes a long time to train, since it takes a lot of time to sample during training.
# 
# Can we make more synthetic images so we can increase the dataset, as in https://arxiv.org/abs/1807.10225?
# 
# <img src="GAN5.png">

# ## Next steps
# Once you've trained your own dataset, convert to the format suitable for the ISLES challenge and upload your own result to participate in the ISLES challenge!
# 
# For final evaluation for submission to ISLES challenge, use the flags:
#     
# `--restore_checkpoints=True`
# 
# `--train_or_test=test`
# 
# This will turn off training so it only does inference to generate results. Also, be sure to set `--test_data_dir_A` to the appropriate directory for the test set for the challenge.
# 
# You can find the code used in this lab at:
# 
# https://github.com/khcs/brain-synthesis-lesion-segmentation/blob/master/utils/merge_2d_test_to_nii.py
# 
# https://github.com/khcs/brain-synthesis-lesion-segmentation/blob/master/utils/convert_3d_test_to_nii.py

# In[ ]:




