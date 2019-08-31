## Actuarial Reserve Risk Classification with Gaussian Mixture
### ARRC



Abstract :
The loss triangle is effective in calculating the actuarial reserve. In this work, we will try to classify each number in this triangle 
into a specific category useful in estimating the risks associated with this.
We will not focus too much on the loss triangle or use an innovative way to create it, but will focus more on making the classification creatively.
The final shape of the loss triangle will be different, as it will be a heat map showing the different ratings for each number,
Also we will use the new data generation from our model as a measure of the expectation of the missing part of the loss triangle.

> You can easily download all the data file from <a href="https://www.dropbox.com/s/fxoqnyzh9rrejcr/final_data4.xlsx?dl=0" target="_blank">**Here**</a> and <a href="https://www.dropbox.com/s/8iw4hxjod84dtsa/final_data3.xlsx?dl=0" target="_blank">**Here**</a> and <a href="https://www.dropbox.com/s/08tcedphnud3gvz/final_data2.xlsx?dl=0" target="_blank">**Here**</a> and the last file is <a href="https://www.dropbox.com/s/y0sqcu82fat5pcl/final_data1.xlsx?dl=0" target="_blank">**Here**</a>

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

Our data not really ideal for using the Gaussian model ! 
But the fun part is that it's like lognormal so we can convert it to normal or at least like normal using log transfrom 

After applying log transform we have : 

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

# this our most important list 
groupby_list=['IncurredAgeBucket','CalYear','ClaimType','ClaimDuration','Gender']  

```
> ###### The idea is we will manipulate the classification of our target several times and make a set of models,this will make a set of loss triangles
We have big data and we don't know how to choose it in principle but we will loop it with fixed classifications that will be necessary to maintain chainladder rules.

```shell
#import data and use concat from pands to put all data file together
sorted_data=data.sort_values(['IncurredAgeBucket']).reset_index(drop=True) #sort and fix indixing

#This will help us to keep this class in our data 
#(Groupby function, excludes classifications if they are not inside the function, or deal with them in bad way)
groupby_=sorted_data.groupby([ groupby_list[0],groupby_list[1],groupby_list[2],
                              groupby_list[3],groupby_list[4]])
groupby_data=groupby_.sum().reset_index()
                                       #sum using groupby

```

Here we will replace the orignal name of this labels to make it more easy for us
```shell
by_data = pd.DataFrame(np.array(groupby_data))
groupby_data['IncurredAgeBucket'].replace(to_replace=['0-49'], value=int(0),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['50'], value=int(1),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['55'], value=int(2),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['60'], value=int(3),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['65'], value=int(4),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['70'], value=int(5),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['75'], value=int(6),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['80'], value=int(7),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['85'], value=int(8),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['90'], value=int(9),inplace=True)
groupby_data['IncurredAgeBucket'].replace(to_replace=['90+'], value=int(10),inplace=True)

groupby_data['Gender'].replace(to_replace=['M'], value=int(0),inplace=True)
groupby_data['Gender'].replace(to_replace=['F'], value=int(1),inplace=True)

groupby_data['ClaimType'].replace(to_replace=['ALF'], value=int(0),inplace=True)
groupby_data['ClaimType'].replace(to_replace=['HHC'], value=int(1),inplace=True)
groupby_data['ClaimType'].replace(to_replace=['NH'], value=int(2),inplace=True)
groupby_data['ClaimType'].replace(to_replace=['Unk'], value=int(3),inplace=True)
```

here we will use histogram to check our data shape

```shell
fig, ax = plt.subplots()
groupby_data['AmountPaid'].plot.kde(ax=ax, legend=False, title='Histogram: check normallaty ')
groupby_data['AmountPaid'].plot.hist(ax=ax)
y=stats.anderson(groupby_data['AmountPaid'], dist='norm') #sample estimates normallity test
if y[1][2] <0.05 :#check normality
    print('its follow normal')
elif y[1][2] > 0.05 :
    print('no its not normal')
#as we see above our data may(may not) follow normal distribution but it not in ideal shape, so we can do log transform 
#before using GP model 
#one reason that we decide to use log transform is that Histogram give us kind of lognormal shape
```

applying log transform and plot one more time 

```shell
#In this code part and next code we will explain the way of ARRC work and dealing with G-mixtuer 
#So you can find the full documentation in the py file (You will find some part of this code inside big loop)
# Transform part using lN(x+1) if we use in future any predictive method we can reverse it using e^(x-1)
datatrans1=np.log1p(groupby_data['AmountPaid'])
new_=groupby_data[groupby_list[nu*2]].astype(np.int32) # nu here its the loop counter
datatrans=np.array([new_,datatrans1]).T # here we merge labels with data after log transform 


#check avelabilty to apply G-mixtuer model
fig, ax = plt.subplots()
pd.DataFrame(datatrans).plot.kde(ax=ax, legend=False, title='Histogram:check new shape for GP after log  ')
pd.DataFrame(datatrans).plot.hist(ax=ax)
```
Here we will start littel machine learning work we will break the data into two parts. The goal of this is to make one part test for the other.

```shell
# Break up the dataset into X train (75%) and X test (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(datatrans,np.full(np.shape(datatrans)[0],1))))
#apply test and train to our data
X_train = datatrans[train_index]
X_test = datatrans[test_index]

```
As we know we are looking for target to class our data with supervise-learning the best job that our loop do for us is to test more than one label class that already in our data (you can see py file for more info)

let's code ! 

```shell
# here other intrsting part 
#find the best mixture number using bic with different covariances type
optimal=[]
for s in range (2,6):
    for cov_type in ['spherical', 'diag', 'tied', 'full'] :
            models=GaussianMixture(n_components=s, covariance_type=cov_type,max_iter=150,n_init=20
                                   ,random_state=500).fit(X_train)
            bic=models.bic(datatrans)# bic number
            optimal.append([bic,cov_type,s])
```
this is the equition of bic 
` BIC = −2 lnL + k ln (n) `
wehre L is the maximized value of the likelihood function for the estimated model ,k is the number of free parameters , n number of observations 
Note that : you can here use AIC but since AIC dealing with small size sample and we have big sample size so i prefer to use BIC


We can plot this now to see the best mixture number for random sample 
<img src="https://www.researchgate.net/profile/Soo_Kim12/publication/6747415/figure/fig5/AS:277829540433934@1443251132875/BIC-values-corresponding-to-the-number-of-mixture-components-of-the-GMM-to-describe-the.png" title="equation" alt="equation"></a>

