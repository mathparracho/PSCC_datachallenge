# PSCC_datachallenge
A repository with all the ressource needed to participate to the 2023 PSCC data challenge

Welcome to the 2023 PSCC data challenge repository!
Here you will find all the informations and files needed to participate to the 2023 PSCC data challenge.

## bout 2023 PSCC data challenge
from Kit's23 homepage:
"Kidney cancer is diagnosed in more than 430,000 individuals each year, accounting for roughly 180,000 deaths. Kidney tumors are found in an even greater number each year, and in most circumstances, it's not currently possible to radiographically determine whether a given tumor is malignant or benign. Even among tumors presumed to be malignant, many appear to be slow-growing and indolent in nature, which has given rise to "active surveillance" as an increasingly popular management strategy for small renal masses. That said, the progression to metastatic disease is a very serious concern, and there is a significant unmet need for systems that can objectively and reliably characterize kidney tumor images for stratifying risk and for pedicting treatment outcomes."

Kidney cancer is a health, financial and sociological issue of great importance. This is why we have decided to use data from the [Kits 2023](https://kits-challenge.org/kits23/) data challenge under [CC BY-NC-SA licence](https://creativecommons.org/licenses/by-nc-sa/4.0/) to make this hackathon.

This hackathon is also the occasion for participants to get used to different aspect of deep learning research end engineering process:
_analysing data
_learning to use common APIs like Pytorch, Tensorflow
_Working on a cluster with a job scheduler
_Managing a continious integration environment, for example with Git

Through this data challenge, participants will therefore be initiate to the daily tasks of a researcher, and could be used as a way to prepare oneself to future professional activities.

## Roadmap
October 20th: start of the inscription phase.

November 8th: end of the inscription phase.

November 9th: Kick-off presentation

November 20th: IDS Télécom Paris computing platform becomes available.
               Submission of models becomes available with test set #1

December 20th-22th: mid challenge reports of the current leaderboard.

January 21st: Submission with test set #2 becomes available.

January 28th: End of the challenge.
              Analysis of the top leaderboard's solutions.

After validation of the prize winning's solution, an award ceremony will be scheduled.

## The Task
TBD

## The data
The KiTS23 cohort includes patients who underwent cryoablation, partial nephrectomy, or radical nephrectomy for suspected renal malignancy between 2010 and 2022 at an M Health Fairview medical center. A retrospective review of these cases was conducted to identify all patients who had undergone a contrast-enhanced preoperative CT scan that includes the entirety of all kidneys.

Each case's most recent contrast-enhanced preoperative scan (in either corticomedullary or nephrogenic phase) was segmented for each instance of the following semantic classes.

Kidney: Includes all parenchyma and the non-adipose tissue within the hilum.

Tumor: Masses found on the kidney that were pre-operatively suspected of being malignant.

Cyst: Kidney masses radiologically (or pathologically, if available) determined to be cysts.

The dataset is composed of 599 cases with 489 allocated to the training set and 110 in the test set. Many of these in the training set were used in previous challenges:

## The participants
Participants will be students from the IP Paris campus.

## Inscriptions
TBD
(If you want to participate to the data challenge, please fill the inscription form at (link). You will then receive login credentials to connect to the workspace)

## Organisation of the challenge
The challenge will be organised in two phases of one month each.
During the first phase, participants will be able to learn

##Licence and attribution
This challenged is based on the Kits 2023 Kidney challenge, which generously put its dataset to our disposal.
You can find their latest scientific contrinution [here](https://arxiv.org/pdf/1912.01054.pdf).
