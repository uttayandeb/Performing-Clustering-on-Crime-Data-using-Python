
##################     Hierarchical Clustering ################

#



import pandas as pd
import matplotlib.pylab as plt 
Crime_Data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Clustering\\New folder\\crime_data.csv")
Crime_Data

### Excluding the first column and only taking the numerical value of the data set
Crime_data1=Crime_Data.iloc[:,1:]
Crime_data1
# Normalization function (when we have both catagorical and non-catagorical datas)
#or normalize the data in the range of 0 and 1
def norm_func1(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# alternative normalization function (Z-transformation)
#normalizing in the range of -2.5 to +2.5
def norm_func(i):                 
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Crime_Data.iloc[:,1:])
df_norm
##or
#df_norm1=norm_func(Crime_data1)
#df_norm1

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)#to know the datatype

import numpy as np
p = np.array(df_norm) # converting into numpy array format 
p        # arrays only sows index numbers not the columns names 




help(linkage)#to know about linkage
#linkage. Performs hierarchical/agglomerative clustering on the condensed distance matrix y. sized vector where n is the number of original observations paired in the distance matrix
z = linkage(df_norm, method="complete",metric="euclidean")#we will calculate the distance between clusters or records by euclidean distance
z



plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram(i.e Bottom-up approach)
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 
h_complete

cluster_labels=pd.Series(h_complete.labels_)

Crime_Data['clust']=cluster_labels # creating a  new column and assigning it to new column 
Crime_Data
Crime_Data = Crime_Data.iloc[:,[5,0,1,2,3,4]]
Crime_Data
Crime_Data.head()

# getting aggregate mean of each cluster
Crime_Data.iloc[:,2:].groupby(Crime_Data.clust).median()

# creating a csv file 
Crime_Data.to_csv("crime_data.csv",encoding="utf-8")











############        K-means clustering              ###################



from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

# Generating random uniform numbers 
X = np.random.uniform(0,1,1000)#creating 1000 random uniform numbers 
X
Y = np.random.uniform(0,1,1000)
df_xy =pd.DataFrame(columns=["X","Y"])#creating a dataframe with columns X and Y

df_xy.X = X

df_xy.Y = Y
df_xy.plot(x="X",y = "Y",kind="scatter")#since it is randoml created so  its distributed uniformlly

###########generating kmeans model with 5 clusters

model1 = KMeans(n_clusters=5).fit(df_xy)
model1.labels_
model1.cluster_centers_
df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)




df_norm.head(10)  # Top 10 rows

###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
Crime_Data['clust']=md # creating a  new column and assigning it to new column 
df_norm.head()

Crime_Data = Crime_Data.iloc[:,[5,0,1,2,3,4]]
Crime_Data
Crime_Data.iloc[:,1:6].groupby(Crime_Data.clust).mean()

Crime_Data.to_csv("crime_data.csv")

