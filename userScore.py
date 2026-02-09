import math
import numpy as np
import matplotlib.pyplot as plt

#----Confidence Weighted Sigmoid Functions----#

# S-Curve Sigmoid
def sigmoid(t):
    return 1 / (1 + math.exp(-t))

# Main func. q0 and n0 are threshold, adjust to change behaviour, k and k2 are curve ramp constants.
def user_score(pos, neg, owners, q0=0.005, n0=10, k=2, k2=2):

    # Total Reviews
    n = pos + neg
    if n <= 0 or owners <= 1:
        return float("nan")

    # Raw User Score (Ratio of Pos to All Reviews)
    p = pos / n
    # Ratio of Reviews to Owners
    q = n / owners

    '''Confident Weight Formula
        First sigmoid - Weighting for review ratio, compared to defined threshold
        Second sigmoid - Weighting for review count, compared to defined threshold'''
    w = sigmoid(k * (math.log10(q) - math.log10(q0))) * \
        sigmoid(k2 * (math.log10(n) - math.log10(n0)))

    return 0.5 + (p - 0.5) * w

def show(game, pos, neg, owners):
    total = pos + neg
    pos_ratio = pos / total
    review_rate = total / owners

    print(f"{game} ({total:,} reviews, {owners:,} owners):")
    print("-------Game score:", user_score(pos, neg, owners))
    print("Positive ratio:", round(pos_ratio, 4))
    print("Review rate: 1 in", round(1 / review_rate))
    print("-" * 50)

print("Sample Tests with Triple A games:")
# Note: Counter Strike 2 displays reviews for CS2 solely on steam, but the webapi fetches the
# reviews for the appid 730 which also includes CS:GO from before migration.
show("Counter-Strike 2", 7_642_084, 1_173_003, 150_000_000)

show("Clair Obscur: Expedition 33", 106_216, 4_816, 3_500_000)

show("Cyberpunk 2077", 713_071, 131_850, 35_000_000)

show("Hollow Knight", 403_641, 12_305, 7_500_000)

show("Red Dead Redemption 2", 676_667, 56_875, 15_000_000)




owners = np.logspace(3, 9, 1000)   # 1k -> 1B owners
cases = [
    (900, 100, "High rating (90%), Low reviews (1000), 1k to 1B Owners"),
    (100, 900, "Low rating (10%), Low reviews (1000), 1k to 1B Owners"),
    (90_000, 10_000, "High rating (90%), High reviews (100,000), 1k to 1B Owners"),
    (10_000, 90_000, "Low rating (10%), High reviews (100,000), 1k to 1B Owners"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (pos, neg, title) in zip(axes.flat, cases):
    scores = [user_score(pos, neg, o) for o in owners]

    ax.plot(owners, scores)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Owners")
    ax.set_ylabel("Adjusted score")
    ax.grid(True)

plt.tight_layout()
plt.show()