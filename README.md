# Structural_State_Translation

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
- blocks.py provides the blocks used in generator and critic (DGCG model= 2 generators + 2 critics)
- config.py provides the configurations used in the model training
- critic.py provides the critic model
- generator.py provides the generator model
- metric.py provides the FID used in the training
- train.py is the file for training the DGCG model and some visualization
- utils.py is only used for gradient penalty used for the critics during the training




