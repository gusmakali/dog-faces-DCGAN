# GAN-DOGS PROJECT

by Liliia Makashova 

**Note**: as it's not possible to add LFS files to an open git repository, plese find the pretrained models and files uploaded on Google Drive, linked below. 

* [gan_128_128_dogs](https://drive.google.com/drive/folders/1QVIBLlsQN1A63LPej3ns-YSeuVaICQ34?usp=sharing) generates 128*128 images. (230 epochs)
* [gan_64_64_dogs](https://drive.google.com/drive/folders/1oD-pJE8hjUNW5qfgdrah7LrfzuKBrblr?usp=sharing) generates 64*64 images. (170 epochs)

## DATA 

To run this file please dowload one of the datasets:
- if you want to see generated only dog faces (defaut version) - dowload this data https://github.com/GuillaumeMougeot/DogFaceNet/releases/.
- if you want to see generated dogs - download this data (this dataset is much larger, expect slower performance) - http://vision.stanford.edu/aditya86/ImageNetDogs/.

Depending on the choice of dataset, please, don't forget to check the name of it in the `data_path` variable if you are not using the dog faces dataset with the defaut name (as downloaded `./after_4_bis/`). The code would open images from all folders in the dataset.  The data folder should be on the same level as the notebook file.

**Please note**: to run you don't need annotation files, just images. 

## PREPROCESSING and DATA LOADING 

No preprocessing apart from cropping and resizing needed (according to the paper referenced https://arxiv.org/pdf/1511.06434.pdf). 

torchvision package is used for dataloading and batching.

## MODEL

DCGAN architecture followed from paper. Report submitted with this repository refers to the above mentioned article extensively. All hyperpamaters as batch_size, betas for optimisers, scaling of images, learning rates, slope for activations were taken from the papers references in the report (GAN-cats, the original parer by Radford and the initial paper on GANs by Goodfellow). The only thing - size of kernel, stride, padding - were made custom for my convenience to follow the sizes. 

Note: the numbers for flattening in `fc` are calcultated by the formula `(input - kerner_size + 2* padding) / stride + 1`. You would need to uncomment the lines in the default version if you want to run with 128*128 images and respectively comment the default ones.

## TRAINING

The training of the submitted verison will create and save a new model with the name `model_evaluation.pt`. I you want you can change the name when calling the `train` function. 

**Note**: the training will start from scratch.

If you want to add more epochs to a pretrained model (submitted along), just add several extra epochs (e.g. 173 to train for 3 epochs more) to the `num_epochs` variable of the saved model and set the `resume_from_saved` function argument to `True`. This will resume the training of the already saved model for the extra amount of epochs that you specified. If you want to do this, please change the name of the model in the call of the `train` function to the one that you chose to continue training. You would also need to change the `model_name` in all pickled files that are used for generating and plotting if you want to run from pretrained (last two cells of the notebook).

**NOTE**: if you want to resume training of the high definition images, please follow the instruction left in comments in cell # 3 (preprocessing), cell # 10 (learning rates) as well as in `Discriminator` and `Generator` cells. The model needed minor changes to produce higher resolution images. The report submitted along explains those steps. The default, uncommented, code runs for generation of 64*64 images. 

## EVALUATION

`see_generated` function will print saved images from the generator training. The default sample size is 24, you can change  it in the `train` function if you want to be able to see more images. `see_generated` will by default print images generated after the last epoch (it's what -1 stands for). If you want to see the development while training you can change the number that corresponds to the number of epochs in the call for this function. Example:

``` python
see_generated(0, dogs) # to see images generated after epoch 0
```

## GENERAL NOTES

While training, the script would save not only the model, but also loss scores over batches and samples for further generations. The naming of the files to save is coherent, e.g. if the current name of the model saved is `model_evaluation.pt`, then samples would be named `model_evaluation_samples.pkl` and losses - `model_evaluation_losses.pkl`.

The script was extended to have a saving mechanism because of constant 'CUDA out of memory' errors and not being able to train for more that 70 epochs in one go. Now it's possible to resume training after the break and continue to accumulate losses and samples. Note, the model and samples are saved every epoch and losses - every batch. 

A note on loss calulation: as in all papers for loss calulation for GANs, the loss of discriminator is the sum of losses over real images and fake ones (`real_loss` + `fake_loss`), and  the loss for genarator is `real_loss` over noise vector (fake images) that are flipped - e.g. torch.ones -  as its job is to fool discriminator. 

So far, as expected the losses graph looks like a zig-zig. The model doesn't converge yet as it was not trained long enough. The generator so far loses the game to trick the discriminator as it is having a higher loss.
