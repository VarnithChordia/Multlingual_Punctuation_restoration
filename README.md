# Multilingual Punctuation Restoration


This is the repository of the work done for multilingual punctuation restoration for [EACL system demonstration](https://2021.eacl.org/calls/demos/). Our demonstrated system can restore punctuations - period(.), comma(,),exclamation(!) and question mark(?) - for five languages - High Resource(English, French and German) and Low Resource Languages(Hindi and Tamil). The models can be available on request.

## Architecture
The underlying language model consists of  (i) M-BERT (ii) BILSTM (iii) NCRF - Neural Conditonal Random Field. This language model is jointly trained with language classifier and text mode classifier - 'Written' & 'Spoken'. The architecture can be understood from the diagram below:

![BERT_ARCHITECTURE](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/PR_architecture.png)

Adding the auxilliary classifers improves the performance of the system by aligning the weights better for the type of input text and language. 

## Dataset
For this paper we considered the following datasets:

1. [EUROPARL Dataset](https://www.statmt.org/europarl/) - Consisted of spoken dialog in multiple languages (translated and non translated) from members of European union parliament in varying chronology.

2. [Webhose](https://webhose.io/?utm_medium=CPC&utm_source=Google&utm_campaign=1200517_WD-Brand-campaign-global&gclid=CjwKCAjw1ej5BRBhEiwAfHyh1Oo_F73bFNOihGRVFEw0dzwyfxqWhZoj5Vw4kjlbFN3GX2-YVcBmiBoC-vkQAvD_BwE) - Consists of the news articles from top sources in English, German and French language.


We cleaned and preprocessed the data to our needs, to label the dataset we used custom tokenizers  to avoid abbreviations,accentmarkers, etc.  We labeled every word in the sequence according to the punctuation following it. We achieved this by converting it into a set of pairs of (token, punctuation) where punctuation is the null punctuation, if there is no punctuation mark following in the text.

##





