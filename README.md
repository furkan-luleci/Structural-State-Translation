# Condition transfer between prestressed bridges using structural state translation for structural health monitoring

This repository contains the codes and the dataset used to transfer the structural condition between two prestressed bridges using *Structural State Translation*, published in a AI in Civil Engineering by Springer. 

Publication: [Condition transfer between prestressed bridges using structural state translation for structural health monitoring](https://link.springer.com/article/10.1007/s43503-023-00016-0)

## The study ##
- This study uses **Structural State Translation** (SST) methodology for condition transfer between two structurally dissimilar prestressed concrete bridges, *Bridge #1* and *Bridge #2*, by translating the state (or condition) of *Bridge #2* to a new state based on the knowledge obtained from *Bridge #1*. 
- A Domain-Generalized Cycle-Generative (DGCG) model is trained on two distinct data domains, *State-H* (healthy) and *State-D* (damaged), acquired from *Bridge #1* in an unsupervised setting, with the cycle-consistent adversarial technique and the Domain Generalization (DG) learning approach implemented.
- The model is used to generalize and transfer its knowledge to *Bridge #2*. In this sense, DGCG translates the condition of *Bridge #2* to the condition that the model learned after training; which is the 50% missing strands, in addition to 10% cross-section loss in the area of the middle girder
- Specifically, in one scenario, *Bridge #2*’s *State-H* is translated to *State-D*; and in another scenario, *Bridge #2*’s *State-D* is translated to *State-H*.
- Finally, the translated bridge states are evaluated by comparing them to the real states based on their modal parameters and Average MMSC (Mean Magnitude-Squared Coherence) values, showing that the translated states are remarkably similar to the real ones.
- The comparison results show a max difference of 1.12% in the bridges' natural frequencies, a difference of 0.28% in their damping ratios, a minimum MAC of 0.923, and an Average MMSC value of 0.947.

## The code ##
- dataset.py provides the loading the dataset
- blocks.py provides the blocks used in generator and critic (DGCG model)
- config.py provides the configurations used in the model training
- critic.py provides the critic model
- generator.py provides the generator model
- metric.py provides the FID used in the training
- train.py is the file for training the DGCG model and some visualization
- utils.py is only used for gradient penalty used for the critics during the training

## The dataset ##
The dataset used in for the SST is created from numeric a bridge deck models as modeled and analyzed in the Finite Element Analysis (FEA) program. 

- First, the bridge decks are modelled in the FEA program.
- Then, they went through with Time History Analysis (THA) after Gaussian noise is applied.
- Subsequently, the acceleration response signals are extracted from the virtual sensor channels placed on each bridge deck model, so as to form the respective acceleration response dataset for each bridge deck state (4 bridge deck state in total).
- The acceleration response signals are extracted from the virtual sensor channels of each bridge deck model for 1024 seconds and 256 Hz, so as to form the respective dataset for each bridge deck state. Each dataset consists of a 15-channel acceleration response signal. The datasets are denoted as *Dataset 1H*, *Dataset 1D*, *Dataset 2H*, and *Dataset 2D*, where the numbers (1 and 2) represent the bridge sequence and the letters (H and D) represent the state of the bridges, with H meaning "healthy" and D meaning "damaged".
- *Bridge#1* is used for training, which are *Dataset 1H*, *Dataset 1D*.
- *Bridge#2* is used for test, which are *Dataset 2H*, *Dataset 2D*.

The train_4096_span1.rar folder includes the undamaged (a0) and damaged (a0) folders, where each folder includes 16-second acceleration response tensors (each tensor has 4096 data points) collected from undamaged and damaged conditions of *Bridge#1*. This folder is used for training.

The test_4096_span2.rar folder includes the undamaged (a0) and damaged (a0) folders, where each folder includes 16-second acceleration response tensors (each tensor has 4096 data points) collected from undamaged and damaged conditions of *Bridge#2*. This folder is used for testing.

## The DGCG model ##

The number of learnable model parameters DGCG model have is 53.7 million. The single DGCG network architecture is shown in the figure below, as there are two of the same networks due to the cycle-consistent adversarial training nature. For instance, one network is responsible for *State-H* to *State-D*, and the other is for *State-D* to *State-H*.

![Picture2](https://github.com/furkan-luleci/Structural_State_Translation/assets/63553991/702a90d0-e0a9-48d8-ba0f-27388b519f3b)




