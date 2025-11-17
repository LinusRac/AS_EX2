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

fig_knn, ax_knn = plt.subplots(1, 1, figsize=(6, 4))
fig_f1, ax_f1 = plt.subplots(1, 1, figsize=(6, 4))

for missing_ratings in [0.25, 0.75]:
    print(f"\033[32mMissing ratings: {int(missing_ratings*100)}%\033[0m")
    trainset, testset = train_test_split(data, test_size=missing_ratings, random_state=0)

    # KNN
    sim_options_KNN = {'name': "pearson",
                    'user_based': True  # compute similarities between users
                    }

    ks = (np.sqrt(2)**np.arange(1, 20)).astype(np.int16)
    mae_s = np.zeros_like(ks).astype(np.float32)
    print("Searching for the best K value that minimizes MAE...")
    for i, k in enumerate(ks):
        print(".", end="", flush=True) # progress indicator
        # prepare user-based KNN for predicting ratings from trainset25
        algo_knn = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
        algo_knn.fit(trainset)

        predictions_KNN = algo_knn.test(testset)

        mae_s[i] = mae(predictions_KNN, verbose=False)

    best_idx = int(np.argmin(mae_s))
    k = int(ks[best_idx])
    print(f"\n[{int(missing_ratings*100)}%] Searched over {len(np.unique(ks))} values of K. Best one was K={k} (MAE={mae_s[best_idx]:.4f})")
    
    ax_knn.plot(ks, mae_s, label = f"{int(missing_ratings*100)}% missing")
    # annotate best K on the plot for this sparsity level
    best_mae = mae_s[best_idx]
    line_color = ax_knn.lines[-1].get_color() if ax_knn.lines else 'red'
    ax_knn.scatter([k], [best_mae], color=line_color, marker='o', edgecolors='black', zorder=5)
    ax_knn.annotate(f"K={k}\nMAE={best_mae:.3f}", (k, best_mae), textcoords='offset points', xytext=(6,-28), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7), arrowprops=dict(arrowstyle='->', lw=0.5))

    algo_knn = KNNWithMeans(k, sim_options=sim_options_KNN, verbose=False)
    algo_knn.fit(trainset)

    predictions_KNN = algo_knn.test(testset)


    # SVD

    algo_svd = SVD(random_state=3)
    algo_svd.fit(trainset)
    predictions_SVD = algo_svd.test(testset)

    predictions_list = [predictions_KNN, predictions_SVD]
    algo_names = [f"{k}-NN", "SVD"]



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ 


    for i, predictions in enumerate(predictions_list):
        print(f"\033[31m{algo_names[i]}\033[0m")

        mae(predictions)

        Ns = np.array([10, 20, 50, 100])
        f1s = np.zeros_like(Ns).astype(np.float32)
        pre_s = np.zeros_like(Ns).astype(np.float32)
        recall_s = np.zeros_like(Ns).astype(np.float32)

        for j, N in enumerate(Ns):

            precisions, recalls = precision_recall_at_n(predictions, n=N, threshold=4) # we consider relevant ratings 4 stars and above

            # Precision and recall can then be averaged over all users
            pre = sum(prec for prec in precisions.values()) / len(precisions)
            recall = sum(rec for rec in recalls.values()) / len(recalls)
            print(f"    N = {N} -- Precision:", pre)
            print(f"    N = {N} -- Recall:", recall)
            f1_val = 2*pre*recall/(pre+recall) if (pre + recall) > 0 else 0.0
            print(f"    N = {N} -- F1:", f1_val)
            print("")

            f1s[j] = f1_val
            pre_s[j] = pre
            recall_s[j] = recall
        
        ax_f1.plot(Ns, f1s, label=f"F1 of {algo_names[i]} at {int(missing_ratings*100)}% missing")

        # for SVD at 75% missing, show a detailed 3-panel plot (Precision/Recall/F1)
        if algo_names[i] == "SVD" and abs(missing_ratings - 0.75) < 1e-9:
            fig_svd, (axp, axr, axf) = plt.subplots(1, 3, figsize=(12, 3.8))
            axp.plot(Ns, pre_s, marker='^', color='#1f77b4')
            axr.plot(Ns, recall_s, marker='s', color='#ff7f0e')
            axf.plot(Ns, f1s, marker='o', color='#2ca02c')

            for axm, title in zip((axp, axr, axf), ("Precision", "Recall", "F1")):
                axm.set_title(f"SVD @ 75% missing — {title}")
                axm.set_xlabel("N (Top-N size)")
                axm.set_ylabel("Score (0-1)")
                axm.set_ylim(0, 1)
                axm.grid(True)

            fig_svd.tight_layout()
            fig_svd.savefig("svd_75_detailed.png", dpi=300)


        

ax_knn.set_title(f"MAE of KNN")
ax_knn.set_xlabel("K")
ax_knn.set_ylabel("MAE")
ax_knn.grid(True)
ax_knn.legend()

ax_f1.set_title("F1 score")
ax_f1.set_xlabel("N")
ax_f1.set_ylabel("F1 score")
ax_f1.grid(True)
ax_f1.legend()
fig_knn.tight_layout()
fig_knn.savefig("mae_knn.png", dpi=300)

fig_f1.tight_layout()
fig_f1.savefig("metrics_topn.png", dpi=300)

plt.show()