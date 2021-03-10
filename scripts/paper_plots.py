import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


stepwise = pd.DataFrame()
stepwise["C=4"] = [3.25491889, 3.12238034, 3.07172287]
stepwise["C=4 std"] = [0.13550876, 0.08793215, 0.07020637]
stepwise["Ensemble C=4"] = [3.05711913, 2.9551087,  2.94774324]
stepwise["Ensemble C=4 std"] = [0.03531162, 0.0212351,  0.01946164]

stepwise["C=1"] = [3.46304812, 3.37414989, 3.39364223]
stepwise["C=1 std"] = [0.05590609, 0.06223677, 0.03840895]
stepwise["Ensemble C=1"] = [3.4280349,  3.33971617, 3.36755064]
stepwise["Ensemble C=1 std"] = [0.01514538, 0.01821746, 0.01195415]

sns.set_theme()
sns.set(font_scale=.8)
#sns.set_context("paper")

plt.style.use('ieee')

plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times New Roman"],  # specify font here
    "font.size":2})          # specify font size here


#plt.errorbar([0, 1, 2], stepwise["Ensemble C=1"], yerr=stepwise["Ensemble C=1 std"]/ 2)
plt.errorbar([0, 1, 2], stepwise["C=1"], yerr=stepwise["C=1 std"]/ 2)
plt.errorbar([0, 1, 2], stepwise["C=4"], yerr=stepwise["C=4 std"]/ 2)
plt.errorbar([0, 1, 2], stepwise["Ensemble C=4"], yerr=stepwise["Ensemble C=4 std"]/ 2)
plt.gca().set_ylim([2.75, 3.75])
plt.gca().set_xticks([0,1,2])
plt.gca().set_xlabel("Data Collection Iteration")
plt.gca().set_ylabel("Average Test NLL")
plt.legend(["C=1", "C=4", "Ens C=4"])
stepwise_plot = plt.gcf()
stepwise_plot.set_size_inches(3.5, 2.0)
plt.show()


percent = pd.DataFrame()
percent["C=1"] = [3.81809929, 3.67878616, 3.48236057, 3.39480343, 3.36547509, 3.27038518,
 3.22464189, 3.21952513, 3.2147747,  3.18287086]
percent["C=1 std"] = [0.21531686, 0.32237045, 0.21386188, 0.16985423, 0.14732254, 0.13725222,
 0.14689189, 0.11917655, 0.09380463, 0.12666734]
percent["C=4"] = [3.68055062, 3.36026994, 3.22923556, 3.08539496, 2.97829089, 2.93189948,
 2.85290316, 2.81848947, 2.77711869, 2.73928725]
percent["C=4 std"] = [0.0931851,  0.11256565, 0.15503851, 0.13826919 ,0.10009687, 0.10236404,
 0.09328638, 0.09228391, 0.08940506, 0.09991016]
percent["Ensemble C=4"] = [3.33587551, 3.08542785, 2.91753972, 2.81261075, 2.76004043, 2.73719966,
 2.68421817, 2.64215866, 2.61320975, 2.5911231 ]
percent["Ensemble C=4 std"] = [0.14342245, 0.11097578, 0.1334517,  0.12530389, 0.10102022, 0.09636468,
 0.09298564, 0.08615939, 0.09726306, 0.09826145]



percentages = np.linspace(0.1, 1.0, 10).tolist()
#plt.errorbar([0, 1, 2], stepwise["Ensemble C=1"], yerr=stepwise["Ensemble C=1 std"]/ 2)
plt.errorbar(percentages, percent["C=1"], yerr=percent["C=1 std"]/ 2)
plt.errorbar(percentages, percent["C=4"], yerr=percent["C=4 std"]/ 2)
plt.errorbar(percentages, percent["Ensemble C=4"], yerr=percent["Ensemble C=4 std"]/ 2)
plt.gca().set_ylim([2.50, 4.00])
plt.gca().set_xticks(percentages)
plt.gca().set_xlabel("Train Data Used (%)")
plt.gca().set_ylabel("Average Test NLL")
plt.legend(["C=1", "C=4", "Ens C=4"])
data_percentage_plot = plt.gcf()
data_percentage_plot.set_size_inches(3.5, 2.0)
pp = PdfPages("sum_plots.pdf")

# Save and remove excess whitespace
pp.savefig(data_percentage_plot, bbox_inches='tight')
#pp.savefig(stepwise_plot, bbox_inches='tight')
pp.close()