## Actuarial reserve modling with Gaussian Mixture




Abstract :
The loss triangle is effective in calculating the actuarial reserve. In this work, we will try to classify each number in this triangle 
into a specific category useful in estimating the risks associated with this.
We will not focus too much on the loss triangle or use an innovative way to create it, but will focus more on making the model creatively.
The final shape of the loss triangle will be different, as it will be a heat map showing the different ratings for each number,
Also we will use the new data generation from our model as a measure of the expectation of the missing part of the loss triangle.


###### start with this beautiful equation

  <img src="https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/math/3e1b1a5eef9c95b3a62ee32069e3e772adabce34.png" title="equation" alt="equation"></a>

where <img src="https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/math/fb6d665bbe0c01fc1af5c5f5fa7df40dc71331d7.png" title="equation" alt="equation"></a> is the mean, <img src="https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/math/d96c898e14704738c2a866adff83537ba4a6b1f4.png" title="equation" alt="equation"></a> the covariance matrix, and k is the dimension of the space where x takes values.

Here we will take much care of k which represents the number of mixes our data have .
This can, of course, be seen with the naked eye by drawing the histogram like :

<img src="https://i.ibb.co/ctmTfZb/rsz-68747470733a2f2f692e737461636b2e696d6775722e636f6d2f583843794d2e706e67.png" title="equation" alt="equation"></a>

From above histogram it is clear that we are facing a mixture of two Gaussian distributions 

