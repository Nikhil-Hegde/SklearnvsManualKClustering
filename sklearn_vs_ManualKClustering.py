import pandas as pd 
import numpy as np 

nba = pd.read_csv("......./Downloads/nba_2013.csv")

#Clustering Analysis 

'''Point Guard for creating scoring opportunities in the team'''
point_guards = nba[nba['pos'] == 'PG']

'''Points per game'''
point_guards['ppg'] = point_guards['pts'] / point_guards['g']

'''Assist Turnover ratio - After removing 0 turnover values'''
point_guards = point_guards[point_guards['tov'] != 0]
point_guards['atr'] = point_guards['ast'] / point_guards['tov']

'''Plotting points per game and assit turnover ratio'''
plt.scatter(point_guards['ppg'], point_guards['atr'], c='y')
plt.title("Point Guards")
plt.xlabel('Points Per Game', fontsize=13)
plt.ylabel('Assist Turnover Ratio', fontsize=13)
plt.show()

'''Creating random centroids for the data'''

num_clusters = 5
# Use numpy's random function to generate a list, length: num_clusters, of indices
random_initial_points = np.random.choice(point_guards.index, size=num_clusters)
# Use the random indices to create the centroids
centroids = point_guards.loc[random_initial_points]

'''Plotting the centroids, in addition to the point_guards'''
plt.scatter(point_guards['ppg'], point_guards['atr'], color ='yellow')
plt.scatter(centroids['ppg'], centroids['atr'], color ='red')
plt.title("Centroids")
plt.xlabel('Points Per Game', fontsize=13)
plt.ylabel('Assist Turnover Ratio', fontsize=13)
plt.show()

'''Function centroids_to_dict takes centroids data frame object creates 
a cluster_id and conerts the ppg and atr values for that centroid into 
a list of coordinates, and adds both the cluster_id and coordinates_list
into the dictionary thats returned.'''

def centroids_to_dict(centroids):
    dictionary = dict()
    # iterating counter we use to generate a cluster_id
    counter = 0

    # iterate a pandas data frame row-wise using .iterrows()
    for index, row in centroids.iterrows():
        coordinates = [row['ppg'], row['atr']]
        dictionary[counter] = coordinates
        counter += 1

    return dictionary

centroids_dict = centroids_to_dict(centroids)

'''Before assigning players to clusters we need a way to compare the 
ppg and atr values of the players with each cluster's centroids'''

import math

def calculate_distance(centroid,player_values):
	root_distance = 0 

	for x in range(0,len(centroid)):
		difference = centroid[x] - player_values[x]
		squared_difference = difference**2
		root_distance += squared_difference

	euclid_distance = math.sqrt(root_distance)
	return euclid_distance

'''Assigning the players to clusters based on Euclidean distance '''

def assign_to_cluster(row):
	lowest_distance = -1
	closest_cluster = -1

	for cluster_id, centroid in centroids_dict_items():
		df_row = [row['ppg'],row['atr']]
		euclidean_distance = calculate_distance(centroid,df_row)

		if lowest_distnace == -1:
			lowest_distance = euclidean_distance
			closest_cluster = cluster_id
		elif euclidean_distance < lowest_distance:
			lowest_distance = euclidean_distance
			closest_cluster = cluster_id
	return closest_cluster

point_guards['cluster'] = point_guards.apply(lambda row:assign_to_cluster(row),axis = 1)

# Visualizing clusters

def visualize_clusters(df, num_clusters):
	colors = ['b','g','r','c','m','y','k']

	for n in range(num_clusters):
		clustered_df = df[df['cluster']==n]
		plt.scatter(clustered_df['ppg'],clustered_df['atr'], c=colors[n-1])
		plt.xlabel('Points Per Game', fontsize=13)
        plt.ylabel('Assist Turnover Ratio', fontsize=13)
    plt.show()

visualize_clusters(point_guards, 5)

# Recalculating the centroids for each cluster

def recalculate_centroids(df):
    new_centroids_dict = dict()
    
    for cluster_id in range(0, num_clusters):
        values_in_cluster = df[df['cluster'] == cluster_id]
        # Calculate new centroid using mean of values in the cluster
        new_centroid = [np.average(values_in_cluster['ppg']), np.average(values_in_cluster['atr'])]
        new_centroids_dict[cluster_id] = new_centroid
    return new_centroids_dict

centroids_dict = recalculate_centroids(point_guards)

# Since we have recalculated the cluster lets reassign the players

point_guards['cluster'] = point_guards.apply(lambda row: assign_to_cluster(row), axis=1)
visualize_clusters(point_guards, num_clusters)

# Recalculate the centroids and shift the clusters

centroids_dict = recalculate_centroids(point_guards)
point_guards['cluster'] = point_guards.apply(lambda row: assign_to_cluster(row), axis=1)
visualize_clusters(point_guards, num_clusters)

# Few points are changing clusters between every iterations

'''
K-Means doesn't cause massive changes in the makeup of clusters between iterations, meaning that it will always converge and become stable
Because K-Means is conservative between iterations, where we pick the initial centroids and how we assign the players to clusters initially matters a lot
'''

# K- Mean using sk-learn 
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(point_guards[['ppg', 'atr']])
point_guards['cluster'] = kmeans.labels_

visualize_clusters(point_guards, num_clusters)

