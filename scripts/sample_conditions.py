import csv
import math
import random
import itertools

from models.util import gen_samples

random.seed(0)
num_conditions = 21
attributions = ["broken", "clumsy", "competent", "confused", "curious", "efficient", "energetic", "focused", "inquisitive","intelligent", "investigative", "lazy", "lost", "reliable", "responsible"]
num_attributions = len(attributions)

trajs_per_hit = 4
attrs_per_hit = 2
# Target seeing a trajectory 7 times in expectation
num_samples = math.ceil(7 * (num_conditions / trajs_per_hit))
with open("input.csv","w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["id" + str(i) for i in range(trajs_per_hit)] + ["aid" + str(i) for i in range(attrs_per_hit)])
    trajs = list(gen_samples(list(range(num_conditions)), num_samples, trajs_per_hit))
    """    trajs = list(gen_samples(list(range(num_conditions))[1:], num_samples, per_hit))
    trajs = [[0] +l for l in trajs]
    [random.shuffle(l) for l in trajs]"""
    attrs = list(gen_samples(attributions, num_samples, attrs_per_hit))
    for traj, attr in zip(trajs, attrs):
        output.writerow(traj + attr)

with open("orderings.csv", "w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["goalAttribution"] + ["id" + str(i) for i in range(trajs_per_hit)])
    for perm in itertools.permutations([4, 8, 9, 10]):
            output.writerow(["curious"] + list(perm))