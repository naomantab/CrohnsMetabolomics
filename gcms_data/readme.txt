This folder contains GCMS data acquired from four different types of clinical sample.

Samples have been taken from healthy controls (CTRL) and patients diagnosed with one of three gastrointestinal conditions:

* Crohn's disease (CD)
* Ulcerative colitis (UC)
* Irritable bowel syndrome (IBS)

Datasets are organised into folders by sample type, as follows:

* blood (BL)
* breath (BR)
* faeces (FA)
* urine (UR)

Within each folder, datasets are organised as Matlab (.MAT) format files. These files are named as follows:

BWG_{sampletype}_{condition}vsCTRL.MAT contains data of the specified sampletype from patients with the condition and from healthy controls

BWG_{sampletype}_{condition}vsALL.MAT contains data from patients with the condition and from all other samples of the specified sampletype, regardless of diagnosis.

______
For more details and acknowledgements see https://dx.doi.org/10.1007/s11306-014-0650-1