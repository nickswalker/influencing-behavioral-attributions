import csv
import math
import random
from collections import Counter
import itertools
random.seed(0)
num_conditions = 21
attributions = ["broken", "clumsy", "competent", "confused", "curious", "efficient", "energetic", "focused", "intelligent", "investigative", "lazy", "lost", "reliable", "responsible"]
num_attributions = len(attributions)

def gen_samples(population, n, per_hit):
    pool = set(population)
    for _ in range(n):
        if per_hit > len(pool):
            sample = list(pool)
            pool = set(population)
            while True:
                extra = set(random.sample(pool, per_hit - len(sample)))
                if extra.isdisjoint(sample):
                    sample += list(extra)
                    pool.difference_update(extra)
                    break
        else:
            sample = random.sample(pool, per_hit)
            pool.difference_update(sample)
        random.shuffle(sample)
        yield sample

per_hit = 3
# Target seeing a trajectory 6 times in expectation
num_samples = math.ceil(6 * (num_conditions / per_hit))
with open("input.csv","w") as f:
    output = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    output.writerow(["id"+str(i) for i in range(per_hit)] + ["aid"+str(i) for i in range(per_hit)])
    trajs = list(gen_samples(list(range(num_conditions)), num_samples, 3))
    attrs = list(gen_samples(attributions, num_samples, 3))
    for traj, attr in zip(trajs, attrs):
        output.writerow(traj + attr)

