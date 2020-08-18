# Multilingual Punctuation Restoration


This is the repository of the work done for multilingual punctuation restoration for COLING 2020 system demonstration. Our demonstrated system can restore punctuations - period(.), comma(,),exclamation(!) and question mark(?) - for three languages - English, French and German. 

## Architecture
The underlying language model consists of  (i) M-BERT (ii) BILSTM (iii) CRF - Conditonal Random Field. This language model is jointly trained with language classifier and text mode classifier - 'Written' & 'Spoken'. The architecture can be understood from the diagram below:

![BERT_ARCHITECTURE](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/BERTBILSTMCRFJOINT_6.png)

Adding the auxilliary classifers improves the performance of the system by aligning the weights better for the type of input text and language. 

## Dataset
For this system demonstration we considered two datasets:

1.![EUROPARL Dataset](https://www.statmt.org/europarl/) - Consisted of spoken dialog in multiple language (translated and non translated) from members of European union parliament in varying chronology.

2.![Webhose](https://webhose.io/?utm_medium=CPC&utm_source=Google&utm_campaign=1200517_WD-Brand-campaign-global&gclid=CjwKCAjw1ej5BRBhEiwAfHyh1Oo_F73bFNOihGRVFEw0dzwyfxqWhZoj5Vw4kjlbFN3GX2-YVcBmiBoC-vkQAvD_BwE) - Consists of the news articles from top sources in every language.

We cleaned and preprocessed the data to our needs, to label the dataset we used custom tokenizers  to avoid abbreviations,accentmarkers, etc.  We labeled every word in the sequence according to the punctuation following it. We achieved this by converting it into a set of pairs of (token, punctuation) where punctuation is the null punctuation, if there is no punctuation mark following in the text.


## System design
Our system conists of two parts - (i) Online processing and (ii) Offline processing. During the offline processing the the model is trained using the proposed model and we use dynamic quantization.
It can be described as shown below:

![System_design](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/SYSTEM_DESIGN_2.png)


