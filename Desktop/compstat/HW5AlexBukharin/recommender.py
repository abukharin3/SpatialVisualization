import numpy as np
import pandas as pd

class recommender:
    '''
    Movie recommender class, solves netflix problem via
    collaborative filtering and user based filtering
    '''
    def __init__(self, user_mat, movie_mat):
        self.user_mat = user_mat
        self.movie_mat = movie_mat

    def l2_sim(self, x, y):
        dist = np.mean((x - y) ** 2)
        return np.exp(dist ** 2)

    def l1_sim(self, x, y):
        dist = np.mean(np.abs(x - y))
        return np.exp(dist ** 2)

    def l0_sim(self, x, y):
        vec = x - y
        count = 0
        for i in vec:
            if i != 0:
                count += 1
        count /= len(vec)
        return np.exp(count ** 2)

    def total_user_sim(self, i):
        # Returns the total similarity of all users j to item i
        total = 0
        x = self.user_mat[i]
        for j in range(len(self.user_mat)):
            y = self.user_mat[j]
            total += self.l0_sim(x, y)
        return total

    def user_based_filter(self, i):
        # Returns top 5 movie recommendations for user i
        movie_list = np.zeros(len(self.user_mat[0]))
        x = self.user_mat[i]
        total = self.total_user_sim(i)
        for j in range(len(self.user_mat)):
            y = self.user_mat[j]
            movie_list = movie_list + self.l0_sim(x, y) * y
        return self.top_5((movie_list / total)[14:])

    def top_5(self, movie_ratings):
        movie_ratings = list(movie_ratings)
        top = []
        for x in range(5):
            max_movie = max(movie_ratings)
            index = movie_ratings.index(max_movie)
            top.append((index, max_movie))
            movie_ratings[index] -= 5

        return np.array(top)

    def total_movie_sim(self, i):
        # Returns the total similarity of all item j to item i
        total = 0
        x = self.movie_mat[i]
        for j in range(len(self.movie_mat)):
            y = self.movie_mat[j]
            total += self.l0_sim(x, y)
        return total


    def movie_filter(self, i):
        # Returns top 5 movie recommendations for user i based on item filtering
        user = self.user_mat[i][14:]
        movie_list = []
        # Loop through the movies, find rating for each movie
        for j in range(len(user)):
            total = self.total_movie_sim(j)
            x = self.movie_mat[j]
            movies = np.array(len(self.user_mat[:, j]))
            for k in range(len(x)):
                y = self.movie_mat[j]
                sim = self.l0_sim(x, y)
                movies = movies + sim * self.user_mat[:, j]
            movies /= total
            movie_list.append(movies)

        return self.top_5(np.array(movie_list).T[i])




def non_zero_mean(arr):
    # Given a numpy array this returns an array wih the zeros filled
    # in with the mean
    mean = 0
    num = 0
    for a in arr:
        if a != 0:
            mean += int(a)
            num += 1
    mean = mean / num
    for i in range(len(arr)):
        if arr[i] == 0:
            arr[i] = mean
    return arr

# Load data
genre_path = r"C:\Users\Alexander\Desktop\compstat\Homework5\MovieGenres.csv"
movie_path = r"C:\Users\Alexander\Desktop\compstat\Movie.csv"

# Process Movie Matrix
movies = pd.read_csv(movie_path, header = None)
colors = list(set(np.array(movies)[1]))
genders = ["Male", "Female"]
user_mat = np.array(movies).T[1:]
for i in range(len(user_mat)):
    for j in range(len(user_mat[0])):
        if j == 1:
            user_mat[i][j] = colors.index(user_mat[i][j])
        elif j == 0:
            user_mat[i][j] = genders.index(user_mat[i][j])
        elif j > 13:
            non_zero = non_zero_mean(user_mat[:, j])
            user_mat[i][j] = int(non_zero[i])
        else:
            user_mat[i][j] = int(user_mat[i][j])

genres = pd.read_csv(genre_path, header = None)
genre_list = []
for entry in np.array(genres)[:, 1]:
    for word in entry.split(" "):
        genre_list.append(word.strip(","))

genre_list = list(set(genre_list))

movie_mat = np.zeros([len(np.array(genres)), len(genre_list)])
genres = np.array(genres)
for i in range(len(genres)):
    for j in range(len(genre_list)):
        if genre_list[j] in genres[i]:
            movie_mat[i][j] = 1

my_recommender = recommender(user_mat = user_mat, movie_mat = movie_mat)

names = ["The Shawkshank Redemption", "The Godfather", "The Dark Knight", "The Godfather Part II", "The Lord of the Rings III", "Pulp Fiction", "Schindler's List",
        "The Good, the Bad and the Ugly", "12 Angry Men", "Inception", "Fight Club", "The Lord of the Rings I", "Forrest Group", "Star Wars V the Empire Strikes Back",
    "The Lord of the Rings II", "The Matrix", "Goodfellas", "One Flew over the Cuckoo's Nest", "Seven Samural", "Interstellar"]
# Write data to csv
name_list = []
movies = [[] for i in range(5)]
ratings = [[] for i in range(5)]
for i in range(len(user_mat)):
    name_list.append(i)
    data = my_recommender.movie_filter(i)
    for j in range(5):
        movies[j].append(names[int(data[j][0])])
        ratings[j].append(data[j][1])

movie_dict = {"Name" : name_list, "Movie1": movies[0], "Score1": ratings[0], "Movie2": movies[1], "Score2": ratings[1], "Movie3": movies[2], "Score3": ratings[2],
"Movie4": movies[3], "Score4": ratings[3], "Movie5": movies[4], "Score5": ratings[4]}
movie_f = pd.DataFrame(movie_dict)
movie_f.to_csv("item_based2.csv")



