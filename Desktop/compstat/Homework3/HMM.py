import numpy as np

class HMM():

    '''
    Given a sequence, we use the forward backward algorithm to compute the most
    likely state at each time and we compute the most likely sequence of states
    using the viterbi algorithm

    '''

    def __init__(self, emit_matrix, transfer_matrix, sequence,
        init_distribution = np.array([0.5, 0.5])):
        self.emit_matrix = emit_matrix
        self.transfer_matrix = transfer_matrix
        self.sequence = sequence
        self.init_distribution = init_distribution
        self.num_states = len(init_distribution)
        self.observation = self.create_obs(sequence)
        self.paths = [[[] for j in range(3)] for i in range(19)]

    def create_obs(self, sequence):
        obs = []
        for event in sequence:
            diag = []
            for state in range(self.num_states):
                diag.append(self.emit_matrix[state][event - 1])
            obs.append(np.diag(diag))
        obs = np.array(obs)
        return obs

    def forward(self, t):
        # Calculate the forward probabilities
        if t == 0:
            return self.init_distribution
        else:
            first = np.matmul(self.observation[t - 1], self.transfer_matrix.T)
            not_normalalized = np.matmul(first, self.forward(t - 1))
            norm = np.sum(not_normalalized)
            normalized = not_normalalized / norm
            return normalized

    def backward(self, t):
        # Calculate the backwrards probabilities
        if t == len(self.sequence) + 1:
            return np.ones([self.num_states])
        else:
            first = np.matmul(self.transfer_matrix, self.observation[t - 1])
            not_normalalized = np.matmul(first, self.backward(t + 1))
            norm = np.sum(not_normalalized)
            normalized = not_normalalized / norm
            return normalized

    def smoothing(self, t):
        #Smoothing to find most likely state at any time
        total_prob = sum([self.forward(i) * self.backward(i) for i in range(len(self.sequence) + 1)])
        prob = self.forward(t) * self.backward(t) / total_prob
        print(prob / sum(prob))

    def path_likelihood(self, path):
        # Compute the log likelihood of a given path
        likelihood = 0
        likelihood += -np.log(self.init_distribution[path[0]])
        for i in range(1, len(path)):
            transition_log = -np.log(self.transfer_matrix[path[i - 1], path[i]])
            #print(i)
            emit_log =  - np.log(self.emit_matrix[path[i - 1], self.sequence[i - 1] - 1])
            likelihood += transition_log + emit_log
        return likelihood

    def shortest_path(self, time, state):
        # Find the shortest path of length time ending in state
        if len(self.paths[time][state]) == 0:
            if time == 0:
                return [state]
            else:
                options = []
                min_path = 0
                for i in range(self.num_states):
                    path = self.shortest_path(time - 1, i) + [state]
                    options.append(self.path_likelihood(path))
                best_choice = options.index(min(options))
                best_path = self.shortest_path(time - 1, best_choice) + [state]
            self.paths[time][state] = best_path
            return best_path
        else:
            return self.paths[time][state]


    def viterbi(self):
        if self.path_likelihood(self.shortest_path(len(self.sequence), 1)) < self.path_likelihood(self.shortest_path(len(self.sequence), 0)):
            return self.shortest_path(len(self.sequence), 1)
        else:
            return self.shortest_path(len(self.sequence), 0)








def main():
    # Load data
    sequence = [4,4,5,4,3,6,3,1,6,5,6,6,2,6,5,6,6,6]
    E = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [1/10, 1/10, 1/10, 1/10, 1/10, 1/2]])
    A = np.array([[0.95, 0.05], [0.1, 0.9]])
    a = HMM(E, A, sequence)
    for i in range(len(sequence) + 1):
        a.smoothing(i)


    print(a.viterbi())

if __name__== "__main__":
    main()





