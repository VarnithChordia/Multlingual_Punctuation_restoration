# Multilingual Punctuation Restoration


This is the repository of the work done for multilingual punctuation restoration for [EACL system demonstration](https://2021.eacl.org/calls/demos/). Our demonstrated system can restore punctuations - period(.), comma(,),exclamation(!) and question mark(?) - for five languages - High Resource(English, French and German) and Low Resource Languages(Hindi and Tamil). The models can be available on request.

## Architecture
The underlying language model consists of  (i) M-BERT (ii) BILSTM (iii) CRF - Conditonal Random Field. This language model is jointly trained with language classifier and text mode classifier - 'Written' & 'Spoken'. The architecture can be understood from the diagram below:

![BERT_ARCHITECTURE](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/PR_architecture.png)

Adding the auxilliary classifers improves the performance of the system by aligning the weights better for the type of input text and language. 

## Dataset
For this system demonstration we considered two datasets:

1. [EUROPARL Dataset](https://www.statmt.org/europarl/) - Consisted of spoken dialog in multiple languages (translated and non translated) from members of European union parliament in varying chronology.

2. [Webhose](https://webhose.io/?utm_medium=CPC&utm_source=Google&utm_campaign=1200517_WD-Brand-campaign-global&gclid=CjwKCAjw1ej5BRBhEiwAfHyh1Oo_F73bFNOihGRVFEw0dzwyfxqWhZoj5Vw4kjlbFN3GX2-YVcBmiBoC-vkQAvD_BwE) - Consists of the news articles from top sources in every language.

We cleaned and preprocessed the data to our needs, to label the dataset we used custom tokenizers  to avoid abbreviations,accentmarkers, etc.  We labeled every word in the sequence according to the punctuation following it. We achieved this by converting it into a set of pairs of (token, punctuation) where punctuation is the null punctuation, if there is no punctuation mark following in the text.


## System design
Our system conists of two parts - (i) Online processing and (ii) Offline processing. During the offline processing the the model is trained using the proposed model and we use dynamic quantization to reduce the size of the model and latency. The online processing is for interactive use, where users can pass text for restoring punctuation or can pass new language data that can be used to fine tune or retrain the punctuation restoration language model. Our web interface consists of two tabs - Punctuate and Annotate. Text entered by the user under the input section is passed through prepossessing module which strips punctuation that we intend to replace and performs appropriate tokenization to pass the input text to the trained model. Under the annotate tab a user can upload a text file or enter text manually of a given language and select the text mode from a drop-down

The frontend system is described as shown below:

![frontend_design](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/Front_end.png)

**In the frontend we passed German text and this returned the punctuated stream of text.** T


The  entire system can be understood as below:
![System_design](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/SYSTEM_DESIGN_2.png)




