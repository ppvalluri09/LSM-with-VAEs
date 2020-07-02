# LSM-with-VAEs
Latent Space manipulation using Variational Autoencoders

## Output
![Latent Space Manipulation using VAEs](https://raw.githubusercontent.com/ppvalluri09/LSM-with-VAEs/master/output.gif)

## How it Works
We train the network by using a somewhat modified version of vanilla autoencoders, but with a hidden latent space vector to learn the data distribution with a particular <b>mean</b> and <b>std</b> of dimension equal to our latent space dimensions.

The network is trained using the loss function:-

```L(y, yhat) = BCE(y, yhat) + KL_divergence(mean, logvar)```

## Sampling
After the training is done we simply initialize a random vector whose dimension is equal to the dimension of the latent vector and then decode the <b>z-vector</b> with our VAE model which results in a sample from the learnt distribution. The first five features in this app are set by the user instead of being randomly initialized to have some control on the output of the model.
