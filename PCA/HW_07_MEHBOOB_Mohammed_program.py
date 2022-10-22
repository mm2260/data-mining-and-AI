import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------------------------------------
# ### 1. Load Data:

# Load data recorded for each shopper over 10 visits.
csv = pd.read_csv('HW_CLUSTERING_SHOPPING_CART_v2215H.csv')

# We collect data over 10 visits as a means to reduce noise. 

# ---------------------------------------------------------------------------------------------------------
# ### 2. Compute Covariance Matrix:

# Drop the ID column and store the remaining data in a numpy array for further calculations.
data = csv.drop(columns='ID').to_numpy()

# Compute the covariance matrix for our agglomeration data.
covariance = csv.drop(columns='ID').cov()

covariance.shape

# The np.{linear algebra}.eig function returns two lists: w and v
# w are the eigenvalues.
# v are the normalized eigenvectors. v[:,i] is the i-th eigenvector corresponding to eigenvalue w[i].
eigenvalues, eigenvectors = np.linalg.eig(covariance)

# Zip up the eigenvectors and their corresponding eigenvalues in tuples and store
# them in a list as principal components.
principal_componenets = []
for index, eigenvalue in enumerate(eigenvalues):
    principal_componenets.append( ( eigenvalue, eigenvectors[:,index] ) )


# ---------------------------------------------------------------------------------------------------------
# ### 3. Sort _Eigenvalues_ in terms of highest to lowest absolute value:


sorted_principal_componenets = sorted( principal_componenets, key= lambda pc: np.abs(pc[0]), reverse=True )


# ---------------------------------------------------------------------------------------------------------
# ### 4. Normalize Eigenvalues:

normalized_eigenvalues = eigenvalues / sum(np.abs(eigenvalues)) 

print('Normalized Eigenvalues:\n',normalized_eigenvalues)


# ---------------------------------------------------------------------------------------------------------
# #### _Plot Cumulative Sum of Normalized Eigenvalues_:


plt.figure( figsize=(10,4) )
plt.xlabel('Number of Eigenvalues')
plt.xticks(range(0,21))
plt.ylabel('Cumulative Sum')

# Use numpy to calculate the cumulative sum array for our normalized eigenvalues, and plot it.
# The numpy method does not add the 0 eigenvalue case, so make sure to add that to the graph.
plt.plot( np.append( [0], np.cumsum(normalized_eigenvalues) ), marker='o')
plt.grid()
plt.show()

# ---------------------------------------------------------------------------------------------------------
# ### 5. Print first three Eigenvectors:

for principal_componenet in principal_componenets[:3]:
    print( principal_componenet[0], ':', principal_componenet[1],'\n')

pd.DataFrame( [ pc[1] for pc in principal_componenets[:3] ], columns= csv.drop(columns='ID').columns ).round(1)

# Graph the % Explained Variances to understand the relationship between 
# attributes, PC's, and how much each PC contributes to understanding our data.
plt.figure( figsize=(12,6) )
plt.ylabel('% of Explained Variances')
plt.xticks(range(1,21))
plt.xlabel('Principal Component #')

plt.bar( range(1,21), sorted(normalized_eigenvalues*100, reverse=True), color='lightblue' )
plt.plot( range(1,21), sorted(normalized_eigenvalues*100, reverse=True), marker='o', color='k' )

plt.grid()
plt.show()

# ---------------------------------------------------------------------------------------------------------
# ### 6. Project Original Agglomeration Data Onto First Two Eigenvectors / Principal Componenets

# Package the eigenvectors into a matrix, to be used for projection 
feature_vector_transpose = np.array( [ sorted_principal_componenets[0][1], sorted_principal_componenets[1][1] ] )


# Multiply the feature vector matrix with our agglomeration data to get the projected data.
projected_data = np.transpose(np.matmul( feature_vector_transpose, np.transpose(data) ))


# Plot the data we projected
plt.figure( figsize=(12,10) )
plt.title('Projected Data Scatter Plot')
plt.xlabel('Principal Component-1')
plt.ylabel('Principal Component-2')

plt.scatter( projected_data[:,0], projected_data[:,1], color='b' )
plt.show()

# ---------------------------------------------------------------------------------------------------------
# ### 7. K-Means:

# Calculate the cluster centers using SciKit-Lern's implementation of K-Means clustering.
cluster_centers = KMeans(4).fit(projected_data).cluster_centers_ # << We know there are 4 means, through observing the scatter plot of our data.


print('Cluster Centers (2D):\n',cluster_centers)

# ---------------------------------------------------------------------------------------------------------

pd.DataFrame(cluster_centers, columns=['PC-1','PC-2'])

# Plot the cluster centers onto our projected data scatter plot.

plt.figure( figsize=(12,10) )
plt.title('Projected Data Scatter Plot with Cluster Centers')
plt.xlabel('Principal Component-1')
plt.ylabel('Principal Component-2')
    
plt.scatter( projected_data[:,0], projected_data[:,1], color='b' )
for center in cluster_centers:
    plt.plot( center[0],center[1], color='r', marker='o', markersize=12 )
    
plt.show()


# ---------------------------------------------------------------------------------------------------------

# #### _Projectection onto the first three PC's to obtain 3-dimensional data._

# Repeat projection, but now in 3D for the top-three principal components. 

feature_vector_transpose_3D = np.array( [ sorted_principal_componenets[0][1], sorted_principal_componenets[1][1], sorted_principal_componenets[2][1] ] )
projected_data_3D = np.transpose(np.matmul( feature_vector_transpose_3D, np.transpose(data) ))


# 3-D scatter plot for projected data.

fig = plt.figure( figsize=(12,10) )
ax = plt.axes(projection='3d')
ax.set_title('Projected Data Scatter Plot')

plt.xlabel('Principal Component-1')
plt.ylabel('Principal Component-2')
ax.set_zlabel('Principal Component-3')

ax.scatter3D( projected_data_3D[:,0], projected_data_3D[:,1], projected_data_3D[:,2], color='b' )
plt.show()

# ---------------------------------------------------------------------------------------------------------
# ### 8. Finding Center of Mass for the three clusters:

cluster_centers_3D = KMeans(3).fit(projected_data_3D).cluster_centers_

print('Cluster Centers (3D)\n', cluster_centers_3D)


pd.DataFrame(cluster_centers_3D, columns=['PC-1','PC-2','PC-3'])

# Plot cluster centers in 3D.

fig = plt.figure( figsize=(12,10) )
ax = plt.axes(projection='3d')
ax.set_title('Projected Data Scatter Plot with Cluster Centers')

plt.xlabel('Principal Component-1')
plt.ylabel('Principal Component-2')
ax.set_zlabel('Principal Component-3')

ax.scatter3D( projected_data_3D[:,0], projected_data_3D[:,1], projected_data_3D[:,2], color='b', alpha=0.05 )
for center in cluster_centers_3D:
    ax.scatter3D( center[0], center[1], center[2], marker='X', color='r', s=150 )
    
plt.show()

# ---------------------------------------------------------------------------------------------------------
# ### 9. Re-Projection:

# Multiplying the 2-D cluster centers by the first two principal componenets:
print('Re-Projection (2D)\n\n', np.matmul( cluster_centers, feature_vector_transpose ) )

pd.DataFrame(np.matmul( cluster_centers, feature_vector_transpose ), columns=csv.drop(columns='ID').columns ).round(1)

# Multiplying the 3-D cluster centers by the first three principal componenets:
print('Re-Projection (3D)\n\n',np.matmul( cluster_centers_3D, feature_vector_transpose_3D ) )

pd.DataFrame(np.matmul( cluster_centers_3D, feature_vector_transpose_3D ), columns=csv.drop(columns='ID').columns ).round(1)

