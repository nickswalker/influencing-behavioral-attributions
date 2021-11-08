"""
Makes the input CSV files specifying
experiments
"""
import csv
import math
import random
import itertools

from models.util import gen_samples

# Sample configurations for our data collection HITs
random.seed(0)
num_trajectories = 21
# We'll ask for demonstrations for these adjectives
adjectives = ["broken", "clumsy", "competent", "confused", "curious", "efficient", "energetic", "focused", "inquisitive", "intelligent", "investigative", "lazy", "lost", "reliable", "responsible"]

# They'll see four videos and be asked to give two demos
trajs_per_hit = 4
attrs_per_hit = 2
# Target seeing a trajectory 7 times in expectation
num_samples = math.ceil(7 * (num_trajectories / trajs_per_hit))
with open("data_collection_orderings.csv","w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["id" + str(i) for i in range(trajs_per_hit)] + ["aid" + str(i) for i in range(attrs_per_hit)])
    trajs = list(gen_samples(list(range(num_trajectories)), num_samples, trajs_per_hit))
    """    trajs = list(gen_samples(list(range(num_conditions))[1:], num_samples, per_hit))
    trajs = [[0] +l for l in trajs]
    [random.shuffle(l) for l in trajs]"""
    attrs = list(gen_samples(adjectives, num_samples, attrs_per_hit))
    for traj, attr in zip(trajs, attrs):
        output.writerow(traj + attr)

with open("competence_orderings.csv", "w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["goalAttribution"] + ["id" + str(i) for i in range(trajs_per_hit)])
    for perm in itertools.permutations([0, 1, 2, 3]):
            output.writerow(["competent"] + list(perm))

with open("brokenness_orderings.csv", "w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["goalAttribution"] + ["id" + str(i) for i in range(trajs_per_hit)])
    for perm in itertools.permutations([4, 5, 6, 7]):
            output.writerow(["broken"] + list(perm))

with open("curiosity_orderings.csv", "w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["goalAttribution"] + ["id" + str(i) for i in range(trajs_per_hit)])
    for perm in itertools.permutations([8, 9, 10, 11]):
            output.writerow(["curious"] + list(perm))
