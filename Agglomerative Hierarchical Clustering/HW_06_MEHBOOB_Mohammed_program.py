# @author: Mohammed Mehboob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------------------------------------------------------------------------
# Read the training data from the provided CSV file.
csv = pd.read_csv('HW_CLUSTERING_SHOPPING_CART_v2215H.csv')

csv

# ---------------------------------------------------------------------------------------------------------
# ## Feature selection and rejection using cross-correlation
# ---

# Remove the ID column and store the agglomeration data in a numpy array for computations.
data = csv.drop(columns='ID').to_numpy()

# Calculate the cross-correlation matrix for our data.
cross_correlation = csv.drop(columns='ID').corr()

cross_correlation

# ---------------------------------------------------------------------------------------------------------
# ### Which two attributes are most strongly cross-correlated with each other?

# Remove the diagonal and lower triangle parts of the correlation matrix:
columns = csv.drop(columns='ID').columns
dropped_labels = set()
for index_i, column_i in enumerate(columns):
    for index_j, column_j in enumerate(columns[:index_i+1]):
        dropped_labels.add( (column_i, column_j) )

# Sort by absolute cross correlation:
sorted_absolute_cross_correlations = cross_correlation.abs().unstack().drop( labels = dropped_labels ).sort_values(ascending=False)
print('Most strongly cross-correlated:\n',sorted_absolute_cross_correlations[:1])

# ---------------------------------------------------------------------------------------------------------

print('Veggies/Soda cross-correlation:',cross_correlation['Vegges']['  Soda'])


# > Veggies and Soda are most strongly cross-correlated (negatively).

# ---------------------------------------------------------------------------------------------------------
# ###  What is the cross-correlation coefficient of Chips with cereal?

print('Chips/Cereal cross-correlation:',cross_correlation[' Chips']['Cereal'])

# ---------------------------------------------------------------------------------------------------------
# ###  Which attribute is fish most strongly cross-correlated with? 

cross_correlation['  Fish'].drop('  Fish', axis=0 ).abs().sort_values(ascending=False)[:1] 

print('Fish is most storngly cross-correlated with Chips:', cross_correlation['  Fish'][' Chips'] )

# ---------------------------------------------------------------------------------------------------------
# ###  Which attribute is Veggies most strongly cross-correlated with? 

cross_correlation['Vegges'].drop('Vegges', axis=0 ).abs().sort_values(ascending=False)[:1]

print('Veggies are most strongly cross-correlated with Soda:', cross_correlation['Vegges']['  Soda'])

# ---------------------------------------------------------------------------------------------------------
# ### According to this data, do people usually buy milk and cereal?

cross_correlation['  Milk']['Cereal']

print('Cross-correlation between Milk and Cereal:',cross_correlation['  Milk']['Cereal'])

# ---------------------------------------------------------------------------------------------------------
# ###  Which two attributes are not strongly cross-correlated with anything?

print('Last 15 absolute cross-correlations:\n\n',sorted_absolute_cross_correlations[-15:])


# ---------------------------------------------------------------------------------------------------------
# Display the heatmap for cross-correlations, so as to visualize which variables yield the least cross-correlation coefficients across the board. 
plt.figure( figsize = ( 16, 12 ) )
sns.heatmap( cross_correlation.abs(), cmap='mako', annot=True, mask = np.triu(cross_correlation.abs()) )
plt.show()

# Store the cross-correlation values to visualize with line graphs, 
# so that we may compare them to find out which variables are least important.
salt = cross_correlation['  Salt'].drop('  Salt',axis=0).abs()
fruit = csv.drop(columns='ID').corr()[' Fruit'].drop(' Fruit', axis=0).abs()
beans = csv.drop(columns='ID').corr()[' Beans'].drop(' Beans', axis=0).abs()
cereal = csv.drop(columns='ID').corr()['Cereal'].drop('Cereal', axis=0).abs()
eggs = csv.drop(columns='ID').corr()['  Eggs'].drop('  Eggs', axis=0).abs()

# Plot line graphs for the cross-correlations for the less important variables, to determine which are the least important.
# Also plot the mean values for their cross-correlation coefficients, to gain an understanding of the central tendency.
plt.figure( figsize=(20,10))

ax1 = plt.subplot(2,3,1)
ax1.set_ylim(top=0.03)
plt.plot(salt, label='Salt')
plt.axhline(np.mean(salt), label='Mean', linestyle='--', color='darkorange')
plt.xticks(rotation=90)
plt.legend()

ax2 = plt.subplot(2,3,2)
ax2.set_ylim(top=0.03)
plt.plot(fruit, label='Fruit')
plt.axhline(np.mean(fruit), label='Mean', linestyle='--', color='darkorange')
plt.xticks(rotation=90)
plt.legend()

ax3 = plt.subplot(2,3,3)
ax3.set_ylim(top=0.03)
plt.plot(beans, label='Beans')
plt.axhline(np.mean(beans), label='Mean', linestyle='--', color='darkorange')
plt.xticks(rotation=90)
plt.legend()

ax4 = plt.subplot(2,3,4)
ax4.set_ylim(top=0.03)
plt.plot(eggs, label='Eggs')
plt.axhline(np.mean(eggs), label='Mean', linestyle='--', color='darkorange')
plt.xticks(rotation=90)
plt.legend()

ax5 = plt.subplot(2,3,5)
ax5.set_ylim(top=0.03)
plt.plot(cereal, label='Cereal')
plt.axhline(np.mean(cereal), label='Mean', linestyle='--', color='darkorange')
plt.xticks(rotation=90)
plt.legend()

plt.show()

# ---------------------------------------------------------------------------------------------------------
# ###  If you were to delete two attributes, which would you guess were irrelevant?

print(sorted_absolute_cross_correlations[:4])

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# # Agglomerative Clustering
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

NUM_DIM = len(data[0]) # The number of dimensions for our data.

# Cluster class
'''
Class to make working with Clusters easier. 
Provides method to effectively merge clusters, and keeps a running track 
of the cluster centers based on data added to it or during cluster merging.
Also consists of information regarding the parent clusters.
'''
class Cluster:
    def __init__(self, data, parents, center=None):
        global NUM_DIM
        global CLUSTER_ID
        
        CLUSTER_ID = CLUSTER_ID + 1
        self.id = CLUSTER_ID
        
        self.parents = parents
        self.center = np.zeros(NUM_DIM) if center is None else center
        self.data = []
        for datapoint in data:
            self.add(datapoint, center is None)
    
    def add(self, datapoint, setCenter):
        # 0 elements in data:
        #  center = datapt
        # 1 element in data:
        #  center = center + datapt / 2
        # n elements in data:
        #  center = n*center + 1 / n+1        
        if setCenter:
            self.center = ( len(self.data)*self.center + datapoint ) / ( len(self.data)+1 )
        self.data.append(datapoint)
    
    def __repr__(self):
        return 'ID:' + str(self.id) + '\nCenter:' + str(self.center) + '\nData:' + str(self.data)
        

# ---------------------------------------------------------------------------------------------------------
# #### _Initialize cluster singletons_

CLUSTERS = dict()
CLUSTER_ID = -1

for datapoint in data:
    cluster = Cluster( [datapoint], (None, None) )
    CLUSTERS[cluster.id] = cluster


'''
Calculates the euclidean distance between two points in n-dimensional space.
'''
def distance( clusterA, clusterB ):
    return np.linalg.norm(clusterA.center - clusterB.center)

# ---------------------------------------------------------------------------------------------------------
# #### _Initialize distance matrix_

DISTANCE_MATRIX = dict()

def compute_distance_matrix():
    for index_A, id_A in enumerate( CLUSTERS.keys() ):
        DISTANCE_MATRIX[ id_A ] = dict()
        for index_B, id_B in enumerate( list(CLUSTERS.keys())[index_A+1:] ):
            DISTANCE_MATRIX[ id_A ][ id_B ] = distance( CLUSTERS[id_A], CLUSTERS[id_B] ) 

# The linkage matrix contains data regarding the merging together of cluster, 
# used to construct the dendrogram.
LINKAGE_MATRIX = []

def mergeClusters(clusterA, clusterB):
    global CLUSTERS
    global DISTANCE_MATRIX
    
    # Merge clusters A and B to create a new cluster.
    new_cluster_data = clusterA.data + clusterB.data
    new_cluster_center = ( clusterA.center*len(clusterA.data) + clusterB.center*len(clusterB.data) )/( len(clusterA.data) + len(clusterB.data) )
    new_cluster_parents = ( clusterA.id, clusterB.id )
    new_cluster = Cluster( new_cluster_data, new_cluster_parents, new_cluster_center ) 
    
    # { FOR THE LAST FEW MERGES }
    if len(CLUSTERS) <= 11:        
        # Report the minimum cluster size of clusters being merged.
        print( (clusterA.id, clusterB.id), '-->', (new_cluster.id), '\t|', min( len(clusterA.data), len(clusterB.data) ), '| Distance:', DISTANCE_MATRIX[clusterA.id][clusterB.id] )
        
    # Add merger information to linkage matrix, which will be used to construct the dendogram.
    # Data: [ cluster A ID | cluster B ID | distance between A and B | number of original datapoints in merged cluster ]
    linkage = [ clusterA.id, clusterB.id, DISTANCE_MATRIX[ clusterA.id ][ clusterB.id ], len(new_cluster.data) ]
    LINKAGE_MATRIX.append(linkage)    

    # Remove the row for Cluster A.
    DISTANCE_MATRIX.pop( clusterA.id, None )
    CLUSTERS.pop( clusterA.id, None )
    # Remove the row for Cluster B.
    DISTANCE_MATRIX.pop( clusterB.id, None )
    CLUSTERS.pop( clusterB.id, None )
            
    # Create new row for the new cluster.
    DISTANCE_MATRIX[new_cluster.id] = dict()
    
    # Update the Distance Matrix.
    for cluster_id in CLUSTERS.keys():
        if cluster_id == new_cluster.id:
            continue
        
        # Remove the columns for Cluster A and Cluster B.
        if clusterA.id in DISTANCE_MATRIX[cluster_id]:
            DISTANCE_MATRIX[cluster_id].pop( clusterA.id, None )
        if clusterB.id in DISTANCE_MATRIX[cluster_id]:
            DISTANCE_MATRIX[cluster_id].pop( clusterB.id, None )
        
        # Populate the row for the new cluster.        
        # CENTROID LINKAGE:
        DISTANCE_MATRIX[cluster_id][new_cluster.id] = distance(CLUSTERS[cluster_id], new_cluster )
        
    # Add new cluster to our clusters dictionary
    CLUSTERS[new_cluster.id] = new_cluster
    return new_cluster


'''
Find the two closest clusters (by euclidean distance between cluster centers).
'''
def find_closest_pair():
    global CLUSTERS
    global DISTANCE_MATRIX
    
    min_distance = np.infty
    closest_pair = [ -1, -1 ]
    for index_A, id_A in enumerate( CLUSTERS.keys() ):
        for index_B, id_B in enumerate( list(CLUSTERS.keys())[index_A+1:] ):
            dist = DISTANCE_MATRIX[ id_A ][ id_B ]
            if dist < min_distance:
                min_distance = dist
                closest_pair = [ id_A, id_B ]
    
    return closest_pair


# ---------------------------------------------------------------------------------------------------------

compute_distance_matrix()
PROTOTYPES = []

while len(CLUSTERS) > 1:
    closest_pair = find_closest_pair()
    new_cluster = mergeClusters( CLUSTERS[closest_pair[0]], CLUSTERS[closest_pair[1]] )
    
    if( len(CLUSTERS) == 3 ):
        for cluster in CLUSTERS.values():
            PROTOTYPES.append( (cluster.center, len(cluster.data)) ) 

# ---------------------------------------------------------------------------------------------------------

plt.figure( figsize=(16,5))
plt.title('Dendrogram')
plt.ylabel('Cluster Distance')

dn = dendrogram(LINKAGE_MATRIX, truncate_mode = 'lastp', p=20 )
plt.xticks([])

plt.show()

# ---------------------------------------------------------------------------------------------------------

plt.figure( figsize=(16,5))
plt.title('Dendrogram (Package Generated)')
plt.ylabel('Cluster Distance')

dn = dendrogram( linkage(data, 'centroid'), truncate_mode = 'lastp', p=20 )
plt.xticks([])

plt.show()

# ---------------------------------------------------------------------------------------------------------

print(PROTOTYPES[0], PROTOTYPES[1], PROTOTYPES[2], sep='\n*'+'='*70+'*\n')

prototype_df = pd.DataFrame( np.array([ prototype[0] for prototype in PROTOTYPES]), columns=columns )

prototype_df.round(2)

prototype_df.round()

