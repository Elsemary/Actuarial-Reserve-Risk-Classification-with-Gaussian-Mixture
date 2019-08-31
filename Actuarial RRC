#importing libs.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from scipy.stats import multivariate_normal
from scipy import stats
sns.set_style("white")

#The data was loaded from the Excel file provided by SOA, some columns that we do not need in this analysis were removed
#You can download data from github
os.chdir('C:\\Users\\M\\Desktop\\dataproj')
#our data
data1=pd.read_excel('final_data1.xlsx')
data2=pd.read_excel('final_data2.xlsx')
data3=pd.read_excel('final_data3.xlsx')
data4=pd.read_excel('final_data4.xlsx')

#cleaning part

#how many missing values exist
data1.isnull().sum()
data2.isnull().sum()
data3.isnull().sum()
data4.isnull().sum()

#delete any row con. missing value in col. ('Gender', 'CalYear','ClaimDuration','AmountPaid')
data1.dropna(subset=['Gender', 'CalYear','ClaimDuration','AmountPaid','MaxPaid','ServiceDays','IncurredAgeBucket','ClaimType'],inplace=True)
data2.dropna(subset=['Gender', 'CalYear','ClaimDuration','AmountPaid','MaxPaid','ServiceDays','IncurredAgeBucket','ClaimType'],inplace=True)
data3.dropna(subset=['Gender', 'CalYear','ClaimDuration','AmountPaid','MaxPaid','ServiceDays','IncurredAgeBucket','ClaimType'],inplace=True)
data4.dropna(subset=['Gender', 'CalYear','ClaimDuration','AmountPaid','MaxPaid','ServiceDays','IncurredAgeBucket','ClaimType'],inplace=True)

# add all data together
full_data=pd.concat([data1,data2,data3,data4]).reset_index(drop=True)

#From the logic of the data we find that AgeBucket has a big impact so we have arranged the data accordingly
sorted_data=full_data.sort_values(['IncurredAgeBucket']).reset_index(drop=True) #sort and fix indixing

# convert monthes to years manualy #_# (not the perfect way you can handele it with best way )
sorted_data['ClaimDuration'].replace(to_replace=[1,2,3,4,5,6,7,8,9,10,11,12], value=1,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[13,14,15,16,17,18,19,20,21,22,23,24], value=2,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[25,26,27,28,29,30,31,32,33,34,35,36], value=3,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[37,38,39,40,41,42,43,44,45,46,47,48], value=4,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[49,50,51,52,53,54,55,56,57,58,59,60], value=5,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[61,62,63,64,65,66,67,68,69,70,71,72], value=6,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[73,74,75,76,77,78,79,80,81,82,83,84], value=7,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[85,86,87,88,89,90,91,92,93,94,95,96], value=8,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[97,98,99,100,101,102,103,104,105,106,107,108], value=9,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[109,110,111,112,113,114,115,116,117,118,119,120], value=10,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[121,122,123,124,125,126,127,128,129,130,131,132], value=11,inplace=True)
sorted_data['ClaimDuration'].replace(to_replace=[133,134,135,136,137,138,139,140,141,142,143,144], value=12,inplace=True)



# colors that we will use it for plotting (We will use some of them as needed)
colors = [ 'blue', 'black','gold', 'cornflowerblue','darkorange','navy', 'c','pink', 'orange','red', 'yellow','green','brown', 'grey']
# name of our class that we may chose some of it (We will use some of them as needed)
target_names=np.array(['Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9','Class 10','Class 11'])

#The most important list for us as we rely on it in class data
groupby_list=['IncurredAgeBucket','CalYear','ClaimType','ClaimDuration','Gender']   

# Function that draw out ellipses with different covariances
def make_ellipses(gmm, ax):
    for n, color in enumerate(colors): 
        if n < n_classes  : # Avoid any problem that comes from the full color list (different length)
            if gmm.covariance_type == 'full':
                covariances = gmm.covariances_[n][:2, :2] # Determine the covariance from full covariances type
            elif gmm.covariance_type == 'tied':
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == 'diag':
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == 'spherical':
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)# eigenvectors 
            u = w[0] / np.linalg.norm(w[0]) # make it ready to convert to circles
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                      180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.35)
            ax.add_artist(ell)
            ax.set_aspect('equal', 'datalim')
            
#All of these sites have been developed to take into account the chain-ladder method and also take into account the groupby function ,
#which deletes some classifications if not placed inside the function or deal with it badly

groupby_=sorted_data.groupby([ groupby_list[0],groupby_list[1],groupby_list[2],
                              groupby_list[3],groupby_list[4]])
groupby_data=groupby_.sum().reset_index()
#We did this step to make our target perfect (not the best way to do that !)
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
#We did this step to make our target perfect (not the best way to do that !)
groupby_data['Gender'].replace(to_replace=['M'], value=int(0),inplace=True)
groupby_data['Gender'].replace(to_replace=['F'], value=int(1),inplace=True)
#We did this step to make our target perfect (not the best way to do that !)
groupby_data['ClaimType'].replace(to_replace=['ALF'], value=int(0),inplace=True)
groupby_data['ClaimType'].replace(to_replace=['HHC'], value=int(1),inplace=True)
groupby_data['ClaimType'].replace(to_replace=['NH'], value=int(2),inplace=True)
groupby_data['ClaimType'].replace(to_replace=['Unk'], value=int(3),inplace=True)
    
    
#plot our data 
fig, ax = plt.subplots()
groupby_data['AmountPaid'].plot.kde(ax=ax, legend=False, title='Histogram: check normallaty ')
groupby_data['AmountPaid'].plot.hist(ax=ax)
y=stats.anderson(groupby_data['AmountPaid'], dist='norm') #sample estimates normallity test
if y[1][2] <0.05 :#check normality
    print('its follow normal')
elif y[1][2] > 0.05 :
    print('no its not normal')
        #as we see above our data could follow normal dist. but it Not in ideal shape, so we can do log transform 
        #before using GP model 
        #one reason that we decide to use log transform is that Histogram give us kind of lognormal shape

"""
We will manipulate the classification of target several times and make a set of models,
 To make a best loss triangle class

"""
        
our_best_models=[]
for nu in range (0,3):
    # Transform part using lN(x+1) if we use in future any predictive method we can reverse it using e^(x-1)
    datatrans1=np.log1p(groupby_data['AmountPaid'])
    new_=groupby_data[groupby_list[nu*2]].astype(np.int32)
    datatrans=np.array([new_,datatrans1]).T


    #check avelabilty to apply GP mixtuer model
    fig, ax = plt.subplots()
    pd.DataFrame(datatrans).plot.kde(ax=ax, legend=False, title='Histogram:check new shape for GP after log  ')
    pd.DataFrame(datatrans).plot.hist(ax=ax)

    # Break up the dataset into X train (75%) and X test (25%) sets.
    skf = StratifiedKFold(n_splits=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(datatrans,np.full(np.shape(datatrans)[0],1))))
    #apply test and train to our data
    X_train = datatrans[train_index]
    X_test = datatrans[test_index]


    #Here we find best way to develop our target 

    target=datatrans[:,0]

    # Break up the dataset into non-overlapping training (75%) and testing (25%) sets.
    skf = StratifiedKFold(n_splits=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(datatrans,pd.DataFrame(target))))

# X,Y test and train
    X_train = datatrans[train_index]
    y_train = np.array(target[train_index])
    X_test = datatrans[test_index]
    y_test = target[test_index]
#find the best mixture number using bic with different covariances type
    optimal=[]
    for s in range (2,6):#more number here lead to more traffic because the data size is little big
        for cov_type in ['spherical', 'diag', 'tied', 'full'] :
                models=GaussianMixture(n_components=s, covariance_type=cov_type,max_iter=150,n_init=20
                                       ,random_state=500).fit(X_train)
                bic=models.bic(datatrans)# bic number
                optimal.append([bic,cov_type,s])
    # you can here plot this loop using line chart (but it will make our code more slow)
    
    final_components=min(optimal)#lower bic lead to best fis 

    n_classes = final_components[2]
    reg_=[0.0001,0.008,0.05,0.1,0.2,0.3,0.5]#useing different size to control covariance
    param_=['random','kmeans']
    # Try GMMs using different types of covariances.
    for u in (reg_):
        for param in (param_):
            estimators = {cov_type : GaussianMixture(n_components=n_classes,warm_start=True,
                          covariance_type=cov_type, max_iter=300,n_init=20,reg_covar=u,init_params=param,
                          random_state=np.random.randint(250,80000))
                          for cov_type in ['spherical', 'diag', 'tied', 'full']}

            n_estimators = len(estimators)
        #control size of plot
            plt.figure(figsize=(4 * n_estimators // 1, 7))
            plt.subplots_adjust(bottom=.1, top=0.95, hspace=.25, wspace=.05,
                                       left=.1, right=.99)

            for index, (name, estimator) in enumerate(estimators.items()):
                # Since we have class labels for the training data, we can
                # initialize the GMM parameters in a (supervised manner).
                estimator.means_init = np.array([X_train[y_train == o].mean(axis=0)
                                                for o in range(n_classes)])

                # Train the other parameters using the EM algorithm.
                estimator.fit(X_train)

                h = plt.subplot(2, n_estimators // 2, index + 1)
                make_ellipses(estimator, h)
    
                for n, color in enumerate(colors):
                    if n < n_classes  :# to avoid any length problem
                        data = pd.DataFrame(datatrans)[target==n]#use target
                        plt.scatter(np.array(data[0]),np.array(data[1]),color=color ,marker='x',label=target_names[n])
                # Plot the test data with crosses
                for n, color in enumerate(colors):
                    if n < n_classes  :
                        data = pd.DataFrame(X_test)[y_test == n]
                        plt.scatter(np.array(data[0]),np.array(data[1]) ,marker='x', color=color)

                y_train_pred = estimator.predict(X_train)
                train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100 #simple valitaion method that work with supervised way
                plt.text(0.7,0.9 ,'Train accuracy: %.1f' % train_accuracy ,transform=h.transAxes)

                y_test_pred = estimator.predict(X_test)

                test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
                plt.text( 0.7,0.8,'Test accuracy: %.1f' % test_accuracy,transform=h.transAxes)

                plt.xticks(())
                plt.yticks(())
                plt.title(name)
                if train_accuracy > 50 and test_accuracy > 75 and len(X_train)> 35 : # Minimum requirements for acceptance of the any model

                    loss_t_data=groupby_data #Data processing for loss triangle 
                    predi=estimator.predict(datatrans)
                    loss_t_data['class']=predi #predict label for evry value that we have
                    pre_data = loss_t_data['AmountPaid'].groupby([loss_t_data['CalYear']
                                                                 ,loss_t_data['ClaimDuration']]).sum().reset_index()#sum our data with respect to CalYear & ClaimDuration
                    #as we see above we summing AmountPaid respect to CalYear & ClaimDuration we make the same process for our label value

                    class_for_Y_D=[]
                    for i in range(12):
                        class_for_Y_D.append([])
                    for i in range (12):
                        for s in range (12):
                            class_for_Y_D[i].append([])# create nested list to class our data 
                    #Again, this is not the quick and easy way to make this loop , but I didn't have enough time to make it faster
                    year_list = [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011]
                    Duration_list=[1,2,3,4,5,6,7,8,9,10,11,12]
                    for s in range (12): # 
                        for k in range (12):
                            for i in range (len(loss_t_data)):
                                if loss_t_data.iloc[i]['CalYear']==year_list[s] and loss_t_data.iloc[i]['ClaimDuration']==Duration_list[k]:
                                        y=loss_t_data.iloc[i]['class']
                                        class_for_Y_D[s][k].append(y)
                    #sum our labels and convert it to percentage
                    creat_prob_for_class=[]
                    for o in range (12):
                        for k in range (12):
                            for i in range (n_classes):
                                y=np.sum(np.array(class_for_Y_D[o][k])==i)/len(class_for_Y_D[o][k])
                                creat_prob_for_class.append(y)
                    #putting all the class percentage together in different row
                    class_together=[]
                    for i in range (int(len(creat_prob_for_class)/n_classes)): 
                        m=i*n_classes
                        s=m+n_classes
                        x=np.array([creat_prob_for_class[m:s]])
                        class_together.append(x)
                    #concat it
                    prob_=np.concatenate(class_together).astype(np.float32)
                    try: # here we use try to avoid any problems arising from the issue of different labels per model
                        pre_data['class A']=pd.DataFrame(prob_[:,0])#Data processing to be combined with a loss triangle
                        pre_data['class B']=pd.DataFrame(prob_[:,1])
                        pre_data['class C']=pd.DataFrame(prob_[:,2])
                        pre_data['class D']=pd.DataFrame(prob_[:,3])
                        pre_data['class E']=pd.DataFrame(prob_[:,4])
                        pre_data['class F']=pd.DataFrame(prob_[:,5])
                        pre_data['class G']=pd.DataFrame(prob_[:,6])
                    except:
                        pass
                    try:
                        #We will use the TRARA to make the loss triangle 
                        #(it will not affect if the percentage of each labels rise to more than 1 because it will be consistent with other labels 
                        #and can find the ratio between them whatever the number)
                        
                        pre_data['cumsum']=pre_data['AmountPaid'].groupby(pre_data['CalYear']).cumsum()
                        pre_data['cumsum class A']=pre_data['class A'].groupby(pre_data['CalYear']).cumsum()
                        pre_data['cumsum class B']=pre_data['class B'].groupby(pre_data['CalYear']).cumsum()
                        pre_data['cumsum class C']=pre_data['class C'].groupby(pre_data['CalYear']).cumsum()
                        pre_data['cumsum class D']=pre_data['class D'].groupby(pre_data['CalYear']).cumsum()
                        pre_data['cumsum class E']=pre_data['class E'].groupby(pre_data['CalYear']).cumsum()
                        pre_data['cumsum class F']=pre_data['class F'].groupby(pre_data['CalYear']).cumsum()
                        pre_data['cumsum class G']=pre_data['class G'].groupby(pre_data['CalYear']).cumsum()
                    except:
                        pass
                    #Now we have labels but we don't know each label in them represents any level of risk,
                    #That's why I searched for the most affiliated numbers on a particular label and then sum this numbers to see the level of risk,
                    #and the highest in the assembly of course is the most risky
                    # let's code it again (this loop make what i want but still it's not the best)
                    Risk_description_for_every_class=[]
                    for index_ in range (n_classes):
                        ind_=np.where(np.array(pre_data)[:,index_+3]>.40)
                        le=len(ind_[0])
                        x=sum(pre_data.iloc[ind_[0][[i for i in range(le)]]].iloc[:,2])
                        Risk_description_for_every_class.append([x,index_])
                    try:
                        This_our_risk_class=sorted(Risk_description_for_every_class,reverse=True)
                        print(This_our_risk_class[0][1],'is the High Risk')
                        print(This_our_risk_class[1][1],'is the Second level of Risk')
                        print(This_our_risk_class[2][1],'is the Third level of Risk')
                        print(This_our_risk_class[3][1],'is the Fourth level of Risk')
                        print(This_our_risk_class[4][1],'is the Fifth level of Risk')
                        print(This_our_risk_class[5][1],'is the Sixth level of Risk')
                    except:
                        pass
                    
                    #loss_triangle & loss_Square (chainladder) for Actuarial reserve
                    loss_Square = pd.pivot_table(pre_data, index = ['CalYear'], columns = ['ClaimDuration'], values = ['cumsum'])
                    loss_triangle=np.array(loss_Square)
                    loss_triangle=loss_triangle[::-1]#We have created a loss square because we have a lot of data, 
                    loss_triangle=np.tril(loss_triangle, k=0)#and this will enable anyone who wants to predict this will be validation of their prediction
                    loss_triangle=loss_triangle[::-1]
                    
                    """
                    
                    We have finished, we will have a loss triangle and next to it a list containing the detail of each number in this triangle
                    The detail will be the level of risk from which this number came because this number is the sum of the previous
                     This will give the actuary the ability to know the level of risk it was in this year.
                    
                    """

                    our_best_models.append([loss_triangle,loss_Square,This_our_risk_class,pre_data,datatrans,target,estimator,test_accuracy])
                    our_best_models.append(This_our_risk_class)
                    print(pd.DataFrame(loss_triangle),pre_data,This_our_risk_class)
                else:
                    pass


            plt.legend(scatterpoints=1, loc='lower left', prop=dict(size=10))


            plt.show()
