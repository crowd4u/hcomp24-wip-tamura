import argparse
import random

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB

# CONSTANTS
N_TASKS = 3000                ## the number of tasks (test data records)
RANGE_OF_ERROR_SIZE_1 = 0.25  ## the maximum absolute value of $e_1$
RANGE_OF_ERROR_SIZE_2 = 0.25  ## the maximum absolute value of $e_2$

# Args
parser = argparse.ArgumentParser(description='The generator of human worker responses.')
parser.add_argument('seed', help='The random seed', type=int)
parser.add_argument('c', help='The number of classes of the classification tasks, 2, 4, or 8.', type=int)
parser.add_argument('t', help='The number of tasks per human worker', type=int)
parser.add_argument('r', help='The number of human duplicate task assignments', type=int)
parser.add_argument('output_file', help='The path of the output file', type=str)

# Class
class Worker:
    def __init__(self, worker_id, redundancy_slot):
        self.worker_id = worker_id
        self.redundancy_slot = redundancy_slot
        self.error_direction_x = np.random.rand() * 2 - 1
        self.error_direction_y = np.random.rand() * 2 - 1        
        self.error_tendency = np.random.rand() 

# FUNCTIONS
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_dataset(n_classes):
    df = pd.read_csv(f"./datasets/tasks_{n_classes}classes.csv")
    assert N_TASKS == len(df[df["split"]=="test"])
    assert n_classes == len(df["gt"].unique())
    return df

def generate_worker_params(tasks_per_worker, redundancy):
    assert N_TASKS % tasks_per_worker == 0
    n_workers = N_TASKS // tasks_per_worker * redundancy

    # redundancy_slots object to avoid assigning the same task to the same worker
    redundancy_slots = []
    for r in range(redundancy):
        redundancy_slots.extend([r for _ in range(N_TASKS // tasks_per_worker)])
    random.shuffle(redundancy_slots)

    workers = []
    for worker_id in range(n_workers):
        redundancy_slot = redundancy_slots.pop()
        workers.append(Worker(worker_id, redundancy_slot))
    return workers

def train_worker_models(df):
    ## train $f_{\theta}$ with the training data
    train = df[df["split"]=="train"]
    X_train = np.array([train["x"], train["y"]]).T
    gnb = GaussianNB()
    gnb.fit(X_train, train["gt"].astype(np.int_))
    return gnb

def generate_worker_responses(tasks_per_worker, redundancy, df, workers, clf):
    test = df[df["split"]=="test"]

    # Task assignment randomly
    tasks = [] # redundancy , task_id
    tasks_ids = test["task"].tolist()
    for r in range(redundancy):
        random.shuffle(tasks_ids)
        tasks.append(tasks_ids.copy())
    
    # Generate worker responses
    random.shuffle(workers)
    responses_rows = []
    for worker in workers:
        for n in range(tasks_per_worker):
            task_id = tasks[worker.redundancy_slot].pop()

            row = dict()
            row["worker"] = worker.worker_id
            row["task"] = task_id

            task = test[test["task"]==task_id]
            x = task["x"].values[0]
            y = task["y"].values[0]

            if worker.error_tendency > np.random.rand():
                x += worker.error_direction_x * RANGE_OF_ERROR_SIZE_1
                y += worker.error_direction_y * RANGE_OF_ERROR_SIZE_2
            
            row["label"] = str(clf.predict([[x, y]])[0])
            responses_rows.append(row)
    return pd.DataFrame(responses_rows)

# ENTRY POINT
def main():
    args = parser.parse_args()
    set_seed(args.seed)
    df = load_dataset(args.c)
    workers = generate_worker_params(args.t, args.r)
    clf = train_worker_models(df)
    responses = generate_worker_responses(args.t, args.r, df, workers, clf)
    responses.to_csv(args.output_file, index=False)

if __name__ == '__main__':
    main()
