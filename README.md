## Actuarial reserve Risk classification with Gaussian Mixture




Abstract :
The loss triangle is effective in calculating the actuarial reserve. In this work, we will try to classify each number in this triangle 
into a specific category useful in estimating the risks associated with this.
We will not focus too much on the loss triangle or use an innovative way to create it, but will focus more on making the classification creatively.
The final shape of the loss triangle will be different, as it will be a heat map showing the different ratings for each number,
Also we will use the new data generation from our model as a measure of the expectation of the missing part of the loss triangle.


###### start with this beautiful equation

  <img src="https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/math/3e1b1a5eef9c95b3a62ee32069e3e772adabce34.png" title="equation" alt="equation"></a>

where <img src="https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/math/fb6d665bbe0c01fc1af5c5f5fa7df40dc71331d7.png" title="equation" alt="equation"></a> is the mean, <img src="https://docs.scipy.org/doc/scipy-0.14.0/reference/_images/math/d96c898e14704738c2a866adff83537ba4a6b1f4.png" title="equation" alt="equation"></a> the covariance matrix, and k is the dimension of the space where x takes values.

Here we will take much care of k which represents the number of mixes our data have .
This can, of course, be seen with the naked eye by drawing the histogram like :

<img src="https://i.ibb.co/ctmTfZb/rsz-68747470733a2f2f692e737461636b2e696d6775722e636f6d2f583843794d2e706e67.png" title="equation" alt="equation"></a>

From above histogram it is clear that we are facing a mixture of two Gaussian distributions 

### Some problems with the data
Non common problems faced by many such as missing data and ... etc
> Problem in the form of data when we draw histagram
as we see here 

<img src="https://i.ibb.co/dKnxPy8/11.jpg" title="equation" alt="equation"></a>

our data not really ideal for using the Gaussian model ! 
but the fun part is that it's really like lognormal so we can convert it to normal or at least like normal using log transfrom 

after applying log transform we have : 

<img src="https://i.ibb.co/88NfsbY/12.jpg" title="equation" alt="equation"></a>

Here we see that the data is similar to the mixed Gaussian distribution with 2D *note:The blue curves I put for illustration*

As we thought, the EM algorithm decided that the distribution was 2D mixed 

<a href="https://imgbb.com/"><img src="https://i.ibb.co/dPsWkbH/13.jpg" alt="13" border="0"></a>

Let's start working with some codes:

```shell
# importing libs.

import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from scipy.stats import multivariate_normal

# this our most important part list 

groupby_list=['IncurredAgeBucket','ClaimDuration','CalYear','Gender','ClaimType','GroupIndicator',
              'AttainedAgeBucket','ServiceDays']  

```
> ###### The idea is we will randomly manipulate the classification of data several times and make a set of models,this will make a set of loss triangles
We have big data and we don't know how to classify it in principle but we will do it randomly with some fix classifications that will be necessary to maintain chainladder rules.

```shell
#import data and use concat from pands to put all data file together
sorted_data=data.sort_values(['IncurredAgeBucket']).reset_index(drop=True) #sort and fix indixing

#This will help us inside loop to creat multiple models 
groupby_=sorted_data.groupby([ groupby_list[np.random.randint(0,2)],groupby_list[2],
                                      groupby_list[np.random.randint(3,8)]])
                                      # here our idea to manipulate data classification randomly
                                      
groupby_data=groupby_.sum().reset_index() #sum using groupby

```



