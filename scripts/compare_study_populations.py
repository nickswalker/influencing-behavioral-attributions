"""
Check how many participants returned for our final experiments
"""
from processing.loading import process_turk_files
explorative = ["pilot1", "active1", "active2", "mdn_active1", "mdn_active2"]
evaluation = ["in_competence", "in_curiosity", "in_brokenness", "test_competence", "test_curiosity", "test_brokenness"]
exp_ids  = []
for base in explorative:
    cr, d, o, comparison = process_turk_files(base + ".csv")
    exp_ids += list(o["WorkerId"])

eval_ids = []
for base in evaluation:
    cr, d, o, comparison = process_turk_files(base + ".csv")
    eval_ids += list(o["WorkerId"])

print("")