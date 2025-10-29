from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy, KNNWithMeans
from surprise.accuracy import mae
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt



def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """Return precision and recall at n metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of relevant and recommended items in top n
        n_rel_and_rec = sum(
            (true_r >= threshold)
            for (_, true_r) in user_ratings[:n]
        )

        # Precision@n: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec / n

        # Recall@n: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec / n_rel if n_rel != 0 else 0

    return precisions, recalls


data = Dataset.load_builtin('ml-100k')

# ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 


for missing_ratings in [0.25, 0.75]:
    print(f"\033[32mPercentage of missing ratings: {missing_ratings}\033[0m")
    trainset, testset = train_test_split(data, test_size=missing_ratings, random_state=0)

    # KNN
    sim_options_KNN = {'name': "pearson",
                    'user_based': True  # compute similarities between users
                    }

    ks = (np.sqrt(2)**np.arange(1, 20)).astype(np.int16)
    mae_s = np.zeros_like(ks).astype(np.float32)
    for i, k in enumerate(ks):
        print(f"K={k}", end='\r')
        # prepare user-based KNN for predicting ratings from trainset25
        algo = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
        algo.fit(trainset)

        predictions_KNN = algo.test(testset)

        mae_s[i] = mae(predictions_KNN, verbose=False)
    print("                 ")
    
    plt.plot(ks, mae_s, label = f"{missing_ratings}% missing")
    k = ks[np.argmin(mae_s)]

    algo_best = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
    algo_best.fit(trainset)

    predictions_KNN = algo_best.test(testset)


    # SVD

    algo = SVD(random_state=3)
    algo.fit(trainset)
    predictions_SVD = algo.test(testset)

    predictions_list = [predictions_SVD, predictions_KNN]
    algo_names = ["SVD", f"{k}-NN"]



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 


    for i, predictions in enumerate(predictions_list):
        print(f"\033[31m{algo_names[i]}\033[0m")

        mae(predictions)

        precisions, recalls = precision_recall_at_n(predictions, n=5, threshold=4)

        # Precision and recall can then be averaged over all users
        pre = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        print("Precision:", pre)
        print("Recall:", recall)
        print("F1:", 2*pre*recall/(pre+recall))

plt.title(f"MAE of KNN")
plt.xlabel("K")
plt.ylabel("MAE")
plt.show()