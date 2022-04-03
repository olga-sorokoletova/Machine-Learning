# ML-2020/21: Binary Functions Classification

This is the first homework of the [ML 2020/21 course](https://sites.google.com/diag.uniroma1.it/machine-learning/home2021) at Sapienza University of Rome.

## Task Assignment

**Motivation:** During reversing of an unknown software it is often useful to find specific functions, as an example, encryption functions in a malware. 

**Objective:** Given the dataset with the binary functions belonging to 4 classes (Encryption, Math, Sorting and String Manipulation) to provide a solution for the *Binary Functions Classification problem*, comparing 2 different models. 

**Limitations:** To solve the problem, any ML method, except for Neural Networks, was allowed. 

**Expected Output:** A text file with class predictions for the blind test json file (functions without labels).

## Dataset

A dataset of 14397 functions:
* 2724 are Encryption Functions (label: *Encryption*)
* 3104 are String Manipulation (label: *String*)
* 4064 are Sorting Functions (label: *Sort*)
* 4504 are Math Functions (label: *Math*)

Each file is a JSONL file (each line is a json object). 

Each item is a dictionary {**key1:** value1, **key2:** value2,...}. The keys are:
* **ID:** unique id of each function
* **semantic:** the label of each function in {math, sort, encryption, string}
* **list_asm:** the linear list of assembly instruction of each function
* **cfg:** the control flow graph encoded as a networkx graph

## Approach

The following approach to obtain the solution was considered: two different models were obtained by **two different ways of data pre-processing (feature extraction procedure)**: with and without usage of the control flow graph information. 

The preprocessed datasets were then split into training and validation, and the training dataset was used to train the **SVM classifier with the linear kernel** and regularization parameter equal to 1. The best implemented model was used to generate output on the blind test dataset. 

A detailed description of the implemented solution: how data have been preprocessed, which method/algorithm has been used, which configurations of the method have been tried, description of the evaluation method used, and comments on the obtained results can be found in the [report.pdf](https://github.com/olga-sorokoletova/Machine-Learning/blob/main/Homework%201/report.pdf). 

## Implementation

The Jupiter Notebook with implementation of the aforementioned models is attached as [code.ipynb](https://github.com/olga-sorokoletova/Machine-Learning/blob/main/Homework%201/code.ipynb).

## Results

The results on the blind test are provided in [results.txt](https://github.com/olga-sorokoletova/Machine-Learning/blob/main/Homework%201/results.txt).
