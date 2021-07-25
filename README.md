# Multilingual Punctuation Restoration


This is the repository of the work done for multilingual punctuation restoration for [EACL system demonstration](https://2021.eacl.org/calls/demos/), the link to the paper is available at 
. Our demonstrated system can restore punctuations - period(.), comma(,),exclamation(!) and question mark(?) - for five languages - High Resource(English, French and German) and Low Resource Languages(Hindi and Tamil). The models can be available on request.

## Architecture
The underlying language model consists of  (i) M-BERT (ii) BILSTM (iii) NCRF - Neural Conditonal Random Field. This language model is jointly trained with language classifier and text mode classifier - 'Written' & 'Spoken'. The architecture can be understood from the diagram below:

![BERT_ARCHITECTURE](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/PR_architecture.png)

Adding the auxilliary classifers improves the performance of the system by aligning the weights better for the type of input text and language. 

## Dataset
For this paper we considered the following datasets:

The high resource langauges(English, French and German) were obtained from

1. [EUROPARL Dataset](https://www.statmt.org/europarl/) - Consisted of spoken dialog in multiple languages (translated and non translated) from members of European union parliament in varying chronology.

2. [Webhose](https://webhose.io/?utm_medium=CPC&utm_source=Google&utm_campaign=1200517_WD-Brand-campaign-global&gclid=CjwKCAjw1ej5BRBhEiwAfHyh1Oo_F73bFNOihGRVFEw0dzwyfxqWhZoj5Vw4kjlbFN3GX2-YVcBmiBoC-vkQAvD_BwE) - Consists of the news articles from top sources.

While for the low resource languages 

1. [Press Bureau of India & Prime Minister's speech](http://preon.iiit.ac.in/~jerin/bhasha/) - Consists of spoken and written corpora translated from English to Hindi and Tamil.

We cleaned and preprocessed the data to our needs, to label the dataset we used custom tokenizers  to avoid abbreviations,accentmarkers, etc.  We labeled every word in the sequence according to the punctuation following it. We achieved this by converting it into a set of pairs of (token, punctuation) where punctuation is the null punctuation, if there is no punctuation mark following in the text.


An example of the preprocessed data is as seen below:

![BERT_ARCHITECTURE](https://github.com/VarnithChordia/Multlingual_Punctuation_restoration/blob/master/preprocessed_data_.png)


## Train & Inference

To train the model on a dataset run - 

```
python3 train_model.py
```

The arguments for trianing can be altered in the code, with the documentation provided

To predict, prepare a test dataset similar to the train and run the following -

```
python3 predict_model.py
```




## Model availability

The models are available here can can be available on request - https://drive.google.com/drive/folders/1aJYlpjgmiP9ikZLKHi4uOL7Wen92rPBT?usp=sharing


## Citation

```

@inproceedings{chordia-2021-punktuator,
    title = "{P}un{K}tuator: A Multilingual Punctuation Restoration System for Spoken and Written Text",
    author = "Chordia, Varnith",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-demos.37",
    pages = "312--320",
    abstract = "Text transcripts without punctuation or sentence boundaries are hard to comprehend for both humans and machines. Punctuation marks play a vital role by providing meaning to the sentence and incorrect use or placement of punctuation marks can often alter it. This can impact downstream tasks such as language translation and understanding, pronoun resolution, text summarization, etc. for humans and machines. An automated punctuation restoration (APR) system with minimal human intervention can improve comprehension of text and help users write better. In this paper we describe a multitask modeling approach as a system to restore punctuation in multiple high resource {--} Germanic (English and German), Romanic (French){--} and low resource languages {--} Indo-Aryan (Hindi) Dravidian (Tamil) {--} that does not require extensive knowledge of grammar or syntax of a given language for both spoken and written form of text. For German language and the given Indic based languages this is the first towards restoring punctuation and can serve as a baseline for future work.",
}

```






