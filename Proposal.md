# IFI Master's Project Proposal

## Title
**Developing Multimodal Transformer Approaches to Address Sexism in Social Media**

## Members
Claudia Jurado Santos ， Nilaksan Selliah ， Jiawei Pei

## Supervisor
Lena Jäger, David Reich

## Background

Social Networks are the main platforms for social complaint, activism, etc. Movements like #MeTwoo, #8M or #Time’sUp have spread rapidly. Under the umbrella of social networks, many women all around the world have reported abuses, discriminations and other sexist experiences suffered in real life. Social networks are also contributing to the transmission of sexism and other disrespectful and hateful behaviours. In this context, automatic tools not only may help to detect and alert against sexism behaviours and discourses, but also to estimate how often sexist and abusive situations are found in social media platforms, what forms of sexism are more frequent and how sexism is expressed in these media. This project will contribute to developing applications to detect sexism.


## Goals

In this project, we aim to investigate methods for the automatic identification and characterization of sexism in social media. The study will follow the framework of the EXIST shared task and focus on analyzing sexist content from multiple sources and modalities, since the dataset used in this project is provided by them.

This multimodal design is intended to capture a broader spectrum of social interactions and linguistic/visual patterns, enabling a deeper understanding of how sexism manifests on social platforms.

In addition to addressing the EXIST subtasks, such as binary sexism classification and the categorization of different forms of sexism, we aim to explore several techniques to enhance the performance of multimodal transformers by applying: 

* **Data Augmentation:** Apply techniques such as back-translation, synonym replacement, masked language modeling, and contextual augmentation to enrich the training data, and compare their effectiveness against baseline models for the textual component.
  
* **Multimodal Models:** Develop and evaluate models capable of jointly processing text and images, contrasting their performance with text-only models.
  
* **Image-to-Text:** Extract textual content from memes and treat it as an additional textual modality, enabling unified processing under NLP frameworks.

* **Cross-Dataset Evaluation:** Conduct evaluations across different datasets to assess model robustness and generalizability beyond a single data source.

* **Fine-Tuning Hate Speech Models:** Adapt existing hate speech detection models (e.g., through Low-Rank Adaptation, LoRA) specifically for misogyny detection, and compare them with general-purpose or multimodal models.

* **Preprocessing Techniques:** Systematically evaluate the impact of different preprocessing strategies (e.g., tokenization, normalization, deletion of stopwords) on downstream performance.

With these goals, we intend to improve the robustness, adaptability, and accuracy of AI models when facing different forms, variations, and contexts of sexism.

Some examples of the tasks from the EXIST shared task are provided below for clarification.

## Examples: 

### Subtask 1.1: Sexism Identification in Tweets
This subtask is a **binary classification**. Systems must decide whether a given tweet contains sexist expressions or behaviors (i.e., it is sexist itself, describes a sexist situation, or criticizes a sexist behavior), and classify it into two categories: **YES** or **NO**.

### YES Examples
> “Mujer al volante, tenga cuidado!”  
> “People really try to convince women with little to no ass that they should go out and buy a body. Like bih, I don’t need a fat ass to get a man. Never have.”

### NO Examples
> “Alguien me explica que zorra hace la gente en el cajero que se demora tanto.”  
> "@messyworldorder it’s honestly so embarrassing to watch and they’ll be like “not all white women are like that”"

---

### Subtask 1.2: Sexism Categorization in Tweets
This subtask classifies **sexist tweets** into one or more of the following categories:

### IDEOLOGICAL AND INEQUALITY
Discredits feminism, rejects gender inequality, or portrays men as victims.

> "Mi hermana y mi madre se burlan de mí por defender todo el tiempo los derechos de todos y me acaban de decir feminazi, la completaron."  
> "I think the whole equality thing is getting out of hand. We are different, thats how were made!"

### STEREOTYPING AND DOMINANCE
Expresses false ideas suggesting women are suitable for certain roles or inferior to men.

> "@Paula2R @faber_acuria A las mujeres hay que amarlas…solo eso… Nunca las entenderás."  
> "Most women no longer have the desire or the knowledge to develop a high quality character, even if they wanted to."

### OBJECTIFICATION
Presents women as objects or focuses on physical attributes to fulfill traditional roles.

> "“Pareces una puta con ese pantalón” - Mi hermano de 13 cuando me vio con un pantalón de cuero."  
> "Don’t get married than blame all woman for your poor investment. You should of got a hooker but instead you choose to go get a wedding ring."

### SEXUAL VIOLENCE
Includes sexual suggestions, harassment, or assault.

> "#MeToo Estas 4 no han conseguido su objetivo.El juez estima que se abrieron de patas..."  
> "fuck that cunt, I would with my fist"

### MISOGYNY AND NON-SEXUAL VIOLENCE
Expresses hatred or violence towards women.

> "Las mujeres de hoy en dia te enseñar a querer… estar soltero"  
> "Some woman are so toxic they don’t even know they are draining everyone around them in poison..."

---

### Subtask 2.1: Sexism Identification in Memes
This is a **binary classification** subtask. The system must determine whether a meme **describes a sexist situation or criticizes a sexist behavior**, classifying it as **YES** or **NO**.  
### Examples of Memes

**(a) YES**  
![YES Meme](https://github.com/user-attachments/assets/017344a3-264d-4776-91a8-85e48a9fb92e)

**(b) NO**  
![NO Meme](https://github.com/user-attachments/assets/40c008f5-bb35-4769-bf63-576b9bc37ac9)

### Subtask 2.2: Sexism Categorization in Memes
This task classifies sexist memes using the same categories as Subtask 1.3:

1. **IDEOLOGICAL AND INEQUALITY**  
2. **STEREOTYPING AND DOMINANCE**  
3. **OBJECTIFICATION**  
4. **SEXUAL VIOLENCE**  
5. **MISOGYNY AND NON-SEXUAL VIOLENCE**

The following figures are some examples of categorized memes.

<img width="1063" height="924" alt="image" src="https://github.com/user-attachments/assets/45d6eabd-ef9e-48b5-b16b-bb0969e1ac06" />

---

## DataSets  

For the development of this project, we will use the datasets released in the context of the EXIST Shared Tasks from 2021 to 2024. Below is a summary of their main features:

Corpus size:

* Between the 2021 and 2023 editions, the datasets include a total of **13,897** labeled samples for training and **4,368** samples for testing.
* The corpora consist of tweets annotated in both Spanish and English.

**Exist21/22**

* The task involved both a binary classification (sexist / non-sexist) and a finer-grained categorization into different types of sexism. 
* While the binary labels are relatively balanced, the categorization into specific sexism types shows a strong class imbalance.
* In particular, the non-sexist class is heavily overrepresented compared to the sexist categories, while objectification is the least represented among sexism types.

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/27272a1b-48b4-4a19-a3a9-1835fa91bea2" />

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/42119451-70f2-48bd-804e-64a33355e8c7" />

**EXIST23**
* A new dimension was introduced: user intention classification (e.g., Direct, Judgemental, Reported, etc.).
* Multiple annotators were preserved for each tweet, maintaining diversity of perspectives.
* Tweets were allowed to be assigned multiple labels, which better captures the complexity of sexist discourse in social media.

**EXIST24**

* The dataset was further enriched with annotator demographic variables: education level, age, ethnicity, and country of origin.
* This addition enables research on how annotator diversity affects labeling decisions, providing a more robust resource for studying bias and inter-annotator agreement.

<img width="1589" height="590" alt="image" src="https://github.com/user-attachments/assets/e459e091-36a6-49eb-8adb-6a623d150334" />

More information about the dataset can be found in the [notebook](DataCheck_Summary.ipynb) 


## Models  

For this project, we plan to experiment with both **text-based models** and **meme-based multimodal models**.  

---

### Text-based Tasks

1. **BERT-base-multilingual-cased** [Link](https://huggingface.co/google-bert/bert-base-multilingual-cased)  
   - Pretrained on the top 104 languages with the largest Wikipedia using a **masked language modeling (MLM)** objective.  
   - Supports both English and Spanish, making it suitable as a **baseline model**.  

2. **Hate-speech-CNERG/english-abusive-MuRIL** & **Hate-speech-CNERG/dehatebert-mono-spanish**   [Link](https://huggingface.co/Hate-speech-CNERG)  
   - Specifically trained for **hate speech detection** in English and Spanish.  
   - Achieve strong results on abusive language tasks.  
   - Limitation: struggle with **subtle or implicit sexist content** that does not contain overtly offensive language.  

3. **xlm-roberta-base-misogyny-sexism-indomain-mix-bal**  [Link](https://huggingface.co/annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal)  
   - Trained on **English misogyny/sexism datasets**.  
   - Strong in-domain performance on English sexist content.  
   - Limitation: not trained on **Spanish misogyny corpora** → opportunity for extension.  

4. **Fine-tuned multilingual model**  
   - We will fine-tune **BERT-base-multilingual-cased** on the **EXIST dataset**.  
   - Goal: improve detection of sexism in **both English and Spanish**, especially subtle and context-dependent cases.

5. **RoBERTuito model: Fine-tuned multilingual model**
  - We will fine-tune RoBERTuito, a model pretrained on social media data.
  - Goal: improve detection of sexism in both English and Spanish, taking advantage of the patterns learned from social media during pretraining.

---

### Meme-based Tasks  

1. 1. **CLIP (Contrastive Language-Image Pretraining)** [Link](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)  
   - CLIP uses a **ViT-based image encoder** to process images, making it well-suited for analyzing memes that contain both visual and textual elements.  
   - The model is pretrained on **large-scale web image–text pairs**, many of which include images with embedded text, such as advertisements, logos, and screenshots. This pretraining allows CLIP to **implicitly capture textual cues** within images, even though it is not a dedicated OCR system.  
   - CLIP is capable of recognizing **clear printed text (e.g., standard fonts and typography)** in images and associating it with corresponding semantic embeddings in the same vector space as natural language text. This enables the model to leverage both visual and textual information in **zero-shot or few-shot scenarios** without additional training.  
   - However, CLIP may be less effective with **handwritten text, small fonts, or highly stylized text**, where explicit OCR might be required to extract the textual content accurately.  
   - Overall, CLIP provides a **powerful multimodal representation**, combining image content and text signals, making it suitable for meme classification tasks where text plays a significant role in conveying meaning or sentiment.
  
     <img width="1997" height="747" alt="image" src="https://github.com/user-attachments/assets/d503f361-8db8-49eb-adc6-2c9d8a12ddd0" />



2. **OCR-enhanced extension (optional)**  
   - If CLIP alone does not sufficiently capture embedded text, we will incorporate **OCR** (Optical Character Recognition).  
   - Pipeline:  
     * Use OCR to extract meme text.  
     * Encode extracted text with **BERT/RoBERTa**.  
     * Fuse OCR text embeddings with CLIP’s multimodal embeddings.  
   - Goal: provide a **richer multimodal representation** of memes, improving performance on sexism detection tasks.  

---


## Evaluation

For model evaluation, we will use the F1-score as the primary metric, both for the binary classification task and for the multilabel classification task. The decision threshold will be initially set to 0.5 and subsequently fine-tuned to determine the value that yields the best performance in this specific context.

In addition, we will also report accuracy to provide a complementary perspective on model performance, acknowledging that while accuracy is easy to interpret, it can be less informative in cases of class imbalance.



