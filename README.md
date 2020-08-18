# Multilingual_Punctuation_restoration


This is the repository of the work done for multilingual punctuation restoration for COLING 2020 system demonstration. Our demonstrated system can restore punctuations - period(.), comma(,),exclamation(!) and question mark(?) - for three languages - English, French and German. The underlying language model consists of  (i) M-BERT (ii) BILSTM (iii) CRF - Conditonal Random Field. This language model is jointly trained with language classifier and text mode classifier - 'Written' & 'Spoken'. The architecture can be understood from the diagram below:

![BERT_ARCHITECTURE](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/BERTBILSTMCRFJOINT_6.png)
