# UCMEC applying the law of diminishing marginal uility
This repository is forked from UCMEC.
An overview of this study and other baseline information can be found [here](https://github.com/qlt315/UCMEC-mmWave-Fronthaul).
IPPO and MAPPO algorithms are modified based on [light-mappo](https://github.com/tinyzqh/light_mappo).
Other MADRL algorithms like IQL and MADDPG are simulated based on [epymarl](https://github.com/uoe-agents/epymarl).

## Installation
```
#Clone the code.
git clone https://github.com/2yunsu/UCMEC_marginal_utility.git
```

```
#move to the repository.
cd UCMEC_marginal_utility
```

```
#Create a virtural environment.
conda create -n ucmec python=3.10
```

```
#Install packages.
pip install -r requirments.txt
```

```
#move to the scripts repository
cd scripts
```

```
#run MAPPO
bash train_ucmec_stat_coop.sh

#run IPPO
bash train_ucmec_stat_nocoop.sh

#run LDMU(Ours)
train_ucmec_stat_ldmu.sh
```


