# IFI Master's Project Proposal

## Title
**Optimizing AI Models to Identify Sexism in Social Media**

## Members
Claudia Jurado Santos ， Nilaksan Selliah ， Jiawei Pei

## Supervisor
David Robert Reich

## Background

Social Networks are the main platforms for social complaint, activism, etc. Movements like #MeTwoo, #8M or #Time’sUp have spread rapidly. Under the umbrella of social networks, many women all around the world have reported abuses, discriminations and other sexist experiences suffered in real life. Social networks are also contributing to the transmission of sexism and other disrespectful and hateful behaviours. In this context, automatic tools not only may help to detect and alert against sexism behaviours and discourses, but also to estimate how often sexist and abusive situations are found in social media platforms, what forms of sexism are more frequent and how sexism is expressed in these media. This project will contribute to developing applications to detect sexism.



## Goals

In this project, we aim to investigate methods for the automatic identification and characterization of sexism in social media. The study will follow the framework of the EXIST shared task and focus on analyzing sexist content from multiple sources and modalities, since the dataset used in this project is extracted from them.

This multimodal design is intended to capture a broader spectrum of social interactions and linguistic/visual patterns, enabling a deeper understanding of how sexism manifests and evolves across platforms and formats.

In addition to addressing the EXIST subtasks—such as binary sexism classification and the categorization of different forms of sexism—we aim to explore several techniques to advance AI models for sexism detection:

* **Data Augmentation:** Apply techniques such as back-translation, synonym replacement, masked language modeling, and contextual augmentation to enrich training data, and compare their effectiveness against baseline models.

* **Multimodal Models:** Develop and evaluate models capable of jointly processing text and images, contrasting their performance with text-only and image-only systems.

* **Image-to-Text:** Extract textual content from memes and treat it as an additional textual modality, enabling unified processing under NLP frameworks.

* **Cross-Dataset Evaluation:** Conduct evaluations across different datasets to assess model robustness and generalizability beyond a single data source.

* **Fine-Tuning Hate Speech Models:** Adapt existing hate speech detection models (e.g., through Low-Rank Adaptation, LoRA) specifically for misogyny detection, and compare them with general-purpose or multimodal models.

* **Preprocessing Techniques:** Systematically evaluate the impact of different preprocessing strategies (e.g., tokenization, normalization, image transformations) on downstream performance.

With these goals, we intend to improve the robustness, adaptability, and accuracy of AI models when facing different forms, variations, and contexts of sexism.

Some examples of the tasks from the EXIST shared task are provided below for clarification.



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

_We need to change this , this is just the explanation of one of the datasets but we got around for and they are not all from ticktok, i think it would better to explain only that they are from the shared tasks and that the types of misoginy there are and number of samples... in which lenguage..._

The dataset used in this project was collected from TikTok using Apify’s TikTok Hashtag Scraper tool, focusing on hashtags potentially associated with sexist content. A rigorous manual selection process was conducted to ensure a balanced representation of both positive and negative seed hashtags. In total, 185 Spanish hashtags and 61 English hashtags were selected, ensuring a broad and representative collection of sexist-related content in both languages.

The collected videos were divided into training and test sets following a chronological and author-based partitioning strategy. This approach preserves temporal coherence while preventing data leakage: authors present in the training set were excluded from the test set, reducing the risk of the model learning author-specific patterns and improving generalization capabilities. Furthermore, each hashtag was required to contribute a minimum number of videos, ensuring a more uniform distribution across the dataset. Final video selection was performed randomly, while maintaining temporal distribution to avoid overrepresentation of any particular time period.

The final dataset consists of over 3,000 videos, with the training set containing 2,524 videos (1,524 Spanish and 1,000 English) and the test set containing 674 videos (304 Spanish and 370 English).

<img width="1306" height="527" alt="Screenshot 2025-08-24 at 19 39 26" src="https://github.com/user-attachments/assets/344889c3-223b-46eb-b422-fdc4f09dae5f" />

###  Dataset Structure [I will delete this section this is only the structure of the last dataset]

The data is stored in **JSON format** with multiple records. Each record contains the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier for the record |
| `lang` | Language of the text (`es` for Spanish in the sample) |
| `text` | Text content of the meme |
| `path-memes` | Possible path to the meme image |
| `number-annotators` | Number of annotators (e.g., 6 in the sample) |
| `annotators` | Annotator identifiers or information |
| `gender-annotators` | Gender of annotators |
| `age-annotators` | Age of annotators |
| `ethnicities-annotators` | Ethnicity of annotators |
| `study_levels-annotators` | Education level of annotators |
| `countries-annotators` | Annotators’ country of residence |
| `labels_task4` | Task 4 labels (e.g., "YES", "DIRECT") |
| `labels_task6` | Task 6 labels (e.g., "Ideological-Inequality") |


## Evaluation

Here we forgot yesterday we need to decide which evaluation metric we are gonna select to evaluate all the models and be coherent. Probably wil have to be F1-score since the datasets are not balance.
- **F1-score** is also reported for reference:
  - Binary tasks: positive-class F1  
  - Multi-label tasks: average F1 across all classes  
---

## Extra 
_Here i may add the points that we could made in case , just in case this is not enough workload_




_To be added_
