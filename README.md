# Counterfactual Watch Model (CWM)
The source code and datasets of  ``Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time''

# Contents
- [Contents](#contents)
- [CWM Description](#CWM-description)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)


# [CWM Description](#contents)
In this study, we aim to counteract the duration bias in video recommendation. We propose counterfactual watch time for interpreting the duration bias in video recommendation and point out that the duration bias is caused by the truncation of the user's counterfactual watch time by video duration. A Counterfactual Watch Model (CWM) is then developed, revealing that the counterfactual watch time equals the time users get the maximum benefit from video recommender systems. A cost-based correction function is defined to transform the counterfactual watch time into the user interest, and the unbiased recommendation model can be learned by optimizing a counterfactual likelihood function defined over observed user watch times. 


# [Dataset](#contents)

- [KuaiRand-Pure](https://drive.google.com/file/d/1SHlXpeGnPWD4K88-yfwtZ4LitB7pk34Q/view?usp=sharing)
- [WeChat](https://drive.google.com/file/d/1A59caEs70M6KjAYDdR-SKZwhNZLp3W3C/view?usp=drive_link)


# [Environment Requirements](#contents)

- Hardware（CPU and GPU）
    - Prepare hardware environment with CPU processor and GPU of Nvidia.
- Framework
    - [Pytorch-1.6.0](https://pytorch.org/get-started/previous-versions/)
- Requirements
  - numpy
  - tqdm
  - pandas
  - argparse
  - skit-learn
  - torch==1.6.0


# [Quick Start](#contents)

After installing the source code, you can start data preprocessing, training and evaluation as follows:

- Data preprocessing, Training and Evaluation

  ```shell
  # Data preprocessing, Training and Evaluation
  bash run.sh
  ```


# [Script Description](#contents)

  ```text
  .
  └─CWM
	  ├─README.md               # descriptions of CWM
	  └─src
		├─main.py                 # run the training and evaluation of CWM
		├─prepare_data.py         # preprocess the dataset
		├─train_model2.py         # training recommendation model
		├─utils                   # some utils toolkit
		  ├─__init__.py
		  ├─arg_parser.py         # accept and parse the shell parameter
		  ├─data_warpper.py       # warp the dataset for training
		  ├─early_stop.py         # early stop for obtaining the best model
		  ├─evaluate.py           # evaluating the performance
		  ├─metrics.py            # the metric for evaluating the performance
		  ├─set_seed.py           # setup the random seed
		  ├─summary_dat.py        # filtrating feature and calculating feature dim
		├─preprocessing           # preprocess the dataset
		  ├─__init__.py
		  ├─cal_baseline_label.py # calculating the baseline label
		  ├─cal_ground_truth.py   # calculating the ground truth label
		  ├─pre_kuairand.py       # preprocessing kuairand dataset
		  ├─pre_wechat.py         # preprocessing wechat dataset
		├─models                  # recommendation models
		  ├─__init__.py
		  ├─afi.py                # AutoInt backbone model
		  ├─dcn.py                # DCN backbone model
		  ├─dfm.py                # DeepFM backbone model
		  ├─fm.py                 # FM backbone model
		  ├─xdfm.py               # xDeepFM backbone model
		  ├─loss_func.py          # Loss function of CWM

  ```
