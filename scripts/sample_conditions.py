import csv
import math
import random

from models.util import gen_samples

random.seed(0)
num_conditions = 21
attributions = ["broken", "clumsy", "competent", "confused", "curious", "efficient", "energetic", "focused", "intelligent", "investigative", "lazy", "lost", "reliable", "responsible"]
num_attributions = len(attributions)

per_hit = 3
# Target seeing a trajectory 6 times in expectation
num_samples = math.ceil(6 * (num_conditions / per_hit))
with open("input.csv","w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["id"+str(i) for i in range(per_hit)] + ["aid"+str(i) for i in range(per_hit)])
    trajs = list(gen_samples(list(range(num_conditions)), num_samples, 3))
    """    trajs = list(gen_samples(list(range(num_conditions))[1:], num_samples, per_hit))
    trajs = [[0] +l for l in trajs]
    [random.shuffle(l) for l in trajs]"""
    attrs = list(gen_samples(attributions, num_samples, 3))
    for traj, attr in zip(trajs, attrs):
        output.writerow(traj + attr)

