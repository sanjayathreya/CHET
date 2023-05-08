# Reproducing `CHET` For CS598: Deep Learning for Healthcare (Spring '23)

## Paper and Group Details

- Paper
  - ID: 28
  - Context- aware health event prediction via transition functions on dynamic disease graphs
  - Citation below
  ```text
  @article{Lu2021ContextawareHE,
    title={Context-aware Health Event Prediction via Transition Functions on Dynamic Disease Graphs},
    author={Chang Lu and Tian Han and Yue Ning},
    journal={ArXiv},
    year={2021},
    volume={abs/2112.05195}
  ```
  - Original Code Repository: https://github.com/LuChang-CS/Chet
- Group
  - ID: 130
  - Members:
    - Sanjay Athreya (ssa5@illinois.edu)
    - Srikanth Reddy Pullaihgari (srp10@illinois.edu)

## Project Structure
- `data/`
  - `mimic3/`: mimic3 dataset
    - `encoded/`: encoded data files after part of preprocessing
    - `parsed/`: sampled datasets that contain paitients-admissions and admission-diagnosis codes
    - `raw/`: raw data files ADMISSIONS.csv, DIAGNOSES_ICD.csv PATIENTS.csv
    - `standard/`: train, validation and testing samples 
  - `mimic4/`: mimic4 dataset
    - `encoded/`: encoded data files after part of preprocessing
    - `parsed/`: sampled datasets that contain paitients-admissions and admission-diagnosis codes
    - `raw/`: raw data files admissions.csv, diagnoses_icd.csv patients.csv
    - `standard/`: train, validation and testing samples
  - `params/`: Trained model parameters for mimic3,mimic4, 3 seeds, heart failure and diagnoses prediction tasks
  - `params-ablation1/`: Ablation1 model parameters
  - `params-ablation2.`: Ablation2 model parameters
- `src`
  - `attention.py`: attention models
  - `config.py`: configuration for datasets and tasks 
  - `eval.py`: model evalution module
  - `layers.py`: the various deep neural net layers for baseline model and ablations
  - `metrics.py`: the evaluation metrics
  - `model.py`: the various models including ablations
  - `preprocess.py`: preprocessing module refactored based on pyhealth api
  - `train.py`: module for training models
  - `utils.py`: module of utility functions
  - `Descriptive-Notebook.ipynb`: Descriptive notebook to explain steps involved
- `out`:
  - `output_preprocess.csv` : output file from preprocessing.py
  - `output_training.csv`: output file from train.py for 'base-model'
  - `output_training-ablation1.csv`: output file from train.py for 'ablation1'
  - `output_training-ablation2.csv`: output file from train.py for 'ablation2'
  - `result_ablations_task_h.csv`: final evaluation result file for ablation heart failure prediction
  - `result_ablations_task_m.csv`: final evaluation result file for ablation diagnoses prediction
  - `result_task_h.csv`: final evaluation result file for heart failure prediction
  - `result_task_m.csv`: final evaluation result file for ablation diagnoses prediction
- `ICD10CM_to_ICD9CM.csv`: mapping of ICD10CM codes to ICD9CM codes
- `requirements.txt`: dependent libraries
- `README.md`

After the processing is complete, we get the following statistics:

```bash
# mimic3-carevue
# patient num: 2169
# max admission num: 23
# mean admission num: 2.45
# max code num in an admission: 39
# mean code num in an admission: 10.70

# mimic4 sample
# patient num: 10000
# max admission num: 93
# mean admission num: 3.79
# max code num in an admission: 39
# mean code num in an admission: 13.51
```

## Execution

### Step 1: Environment Setup and dependencies

- We can easily setup the appropriate environment by using google colab.
- It would be recommended to choose a premium GPU (A100) and high memory specification
- Login to colab notebook and run the following commands 

  ```
  !git clone https://github.com/sanjayathreya/cs598dl4h-project
  !pip install requirements.txt
  !mv /content/cs598dl4h-project /content/CHET
  !mkdir -p /content/CHET/data/mimic3/raw/
  !mkdir -p /content/CHET/data/mimic4/raw/
  ```

### Step 2: Obtaining Various raw files 

- Go to https://physionet.org/content/mimic3-carevue/1.4/ to download the MIMIC-III dataset

  ```
  !wget -r -N -c -np --user [account] --ask-password https://physionet.org/content/mimic3-carevue/1.4/
  ```

- Go into the folder and unzip required three files and copy them to the `/content/CHET/data/mimic3/raw/` folder

  ```
  %cd ~/physionet.org/files/mimiciii/1.4
  !gzip -d ADMISSIONS.csv.gz # Admissions information
  !gzip -d DIAGNOSES_ICD.csv.gz  # diagnoses information
  !gzip -d PATIENTS.csv.gz  # patients information
  cp ADMISSIONS.csv PATIENTS.csv DIAGNOSES_ICD.csv /content/CHET/data/mimic3/raw/
  ```

- Go to https://physionet.org/content/mimiciv/2.2/ to download the MIMIC-IV dataset

  ```
  !wget -r -N -c -np --user [account] --ask-password https://physionet.org/content/mimiciv/2.2/
  ```

- Go into the folder and unzip required three files and copy them to the `/content/CHET/data/mimic4/raw/` folder

  ```
  cd ~/physionet.org/files/mimiciii/1.4
  gzip -d admissions.csv.gz # Admissions information
  gzip -d diagnoses_icd.csv.gz  # diagnoses information
  gzip -d patients.csv.gz  # patients information
  cp admissions.csv patients.csv diagnoses_icd.csv /content/CHET/data/mimic4/raw/
  ```

- Copy the ICD10-ICD9 mapping file to user cache folder `/root/.cache/pyhealth/medcode` folder

  ```
  from pyhealth.medcode import CrossMap
  from pyhealth.datasets import MIMIC4Dataset,MIMIC3Dataset
  !cp /content/CHET/ICD10CM_to_ICD9CM.csv /root/.cache/pyhealth/medcode
  ```

### Step 3: Preprocessing 

  ```
  %cd /content/CHET/src/
  !python preprocess.py
  ```

### Step 3: Training 

- train.py also has the type of model to be used i.e. base-model, ablation1, ablation2
- Change the code to select an appropriate model
- Run the following command
  ```
  %cd /content/CHET/src/
  !python train.py
  ```

### Step 3: Evaluate 

- We can run evaluation of base-models on both datasets, for both tasks and ablation studies

  ```
  %cd /content/CHET/src/
  !python eval.py
  !python eval-ablations.py
  ```

## Analysis and Results

- For MIMIC III we have used a different dataset so its not suprising results look different to the ones produced by authors
- Chet references output as published in the paper
- Chet\* refers the the output reproduced by code in this repository 

**Heart Failure prediction task**

|        | MIMICIII    |             | MIMICIV     |       |
| ------ | ----------- | ----------- | ----------- |-------|
| Models | AUC         | F1          | AUC         | F1    |
| Chet   |       86.14 |       73.08 | 90.83       | 71.14 |
| Chet\* |       76.79 |       66.18 |       94.78 | 79.93 |


**Diagnosis prediction task**

|        |             | MIMICIII |        |       | MIMICIV   | 	      |
| ------ | ----------- |----------|--------|-------| --------- |--------|
| Models | W-f1        | R@10     | R@20   | W-f1  | R@10      | R@20   |
| Chet   |       22.63 | 28.64    | 37.87  | 26.35 | 30.28     | 38.69  |
| Chet\* |       16.07 | 25.04    | 33.84  | 25.12 | 30.66     | 39.15  |



Further analysis can be found at the Jupyter Notebook [here](https://github.com/sanjayathreya/cs598dl4h-project/blob/main/src/Descriptive-Notebook.ipynb).

## Credits

- Our work is based off the original implementation code at https://github.com/LuChang-CS/Chet.
- We have adopted several API's from pyhealth module mainly in preprocessing https://pyhealth.readthedocs.io/en/latest/index.html