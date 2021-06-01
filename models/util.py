import random
import sys

from pytorch_lightning.callbacks import ProgressBarBase
from tqdm import tqdm


def gen_samples(population, n, per_hit):
    pool = list(population)
    for _ in range(n):
        if per_hit > len(pool):
            sample = list(pool)
            pool = list(population)
            while True:
                extra = set(random.sample(pool, per_hit - len(sample)))
                if extra.isdisjoint(sample):
                    sample += list(extra)
                    pool = list(set(pool).difference(extra))
                    break
        else:
            sample = random.sample(pool, per_hit)
            pool = list(set(pool).difference(sample))
        random.shuffle(sample)
        yield sample


class LitProgressBar(ProgressBarBase):

    def __init__(self, *args, **kwargs):
        super(LitProgressBar, self).__init__(*args, **kwargs)
        self.process_position = 0
        self.is_disabled = False
        self.current_epoch = 0
        self.overall = tqdm(
            desc='Overall',
            initial=0,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0
        )

    def on_fit_start(self, trainer, pl_module):
        self.overall.total = trainer.max_epochs

    def on_train_epoch_end(self, trainer, pl_module,x):
        self.overall.update(1)
        self.overall.set_postfix(trainer.progress_bar_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        # self.overall.update(1)
        # self.overall.set_postfix(trainer.progress_bar_dict)
        pass