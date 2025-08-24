# IFI Master's Project Proposal

## Title
**EXIST: sEXism Identification in Social neTworks Fifth Shared Task at CLEF 2025**

## Supervisors
David Robert Reich

## Intended Start
Beginning August 2025 



## Background

Social Networks are the main platforms for social complaint, activism, etc. Movements like #MeTwoo, #8M or #Time’sUp have spread rapidly. Under the umbrella of social networks, many women all around the world have reported abuses, discriminations and other sexist experiences suffered in real life. Social networks are also contributing to the transmission of sexism and other disrespectful and hateful behaviours. In this context, automatic tools not only may help to detect and alert against sexism behaviours and discourses, but also to estimate how often sexist and abusive situations are found in social media platforms, what forms of sexism are more frequent and how sexism is expressed in these media. This Lab will contribute to developing applications to detect sexism.



## Goals

In this project, we aim to investigate methods for the automatic identification and characterization of sexism in social media. The study will follow the framework of the EXIST shared task and focus on analyzing sexist content from multiple sources and modalities.

Specifically, the lab defines nine subtasks distributed across two languages (English and Spanish). These correspond to the same three core tasks—(1) sexism identification, (2) source intention detection, and (3) sexism categorization—applied to three distinct types of media:

Text (tweets)

Images (memes)

Videos (TikToks)

This multimodal design is intended to capture a wider range of social interactions and linguistic/visual patterns, enabling a deeper understanding of how sexism manifests and evolves across platforms and formats.

A condensed schema of the tasks included in the lab is presented in the following table:

<img width="989" height="572" alt="image" src="https://github.com/user-attachments/assets/661d3779-109b-4f43-8da3-1620867c9b13" />


For a more detailed description of each subtask, as well as some examples, check the sections below.

### Subtask 1.1: Sexism Identification in Tweets
This subtask is a **binary classification**. Systems must decide whether a given tweet contains sexist expressions or behaviors (i.e., it is sexist itself, describes a sexist situation, or criticizes a sexist behavior), and classify it into two categories: **YES** or **NO**.

### YES Examples
> “Mujer al volante, tenga cuidado!”  
> “People really try to convince women with little to no ass that they should go out and buy a body. Like bih, I don’t need a fat ass to get a man. Never have.”

### NO Examples
> “Alguien me explica que zorra hace la gente en el cajero que se demora tanto.”  
> "@messyworldorder it’s honestly so embarrassing to watch and they’ll be like “not all white women are like that”"

---

### Subtask 1.2: Source Intention in Tweets
Once a message has been classified as sexist, this subtask categorizes the message according to the **author's intention**, providing insight into the role of social networks in disseminating sexist messages. This is a **ternary classification**:

### DIRECT
The message itself is sexist or incites sexism.

> "A woman needs love, to fill the fridge, if a man can give this to her in return for her services (housework, cooking, etc), I don’t see what else she needs."  
> "Women shouldn’t code…perhaps be influencer/creator instead…it’s their natural strength."

### REPORTED
The message reports or shares a sexist situation.

> "I doze in the subway, I open my eyes feeling something weird: the hand of the man sat next to me on my leg #SquealOnYourPig."  
> "Today, one of my year 1 class pupils could not believe he’d lost a race against a girl."

### JUDGEMENTAL
The message judges or condemns sexist situations.

> "As usual, the woman was the one quitting her job for the family’s welfare…"  
> "21st century and we are still earning 25% less than men #Idonotrenounce."

---

### Subtask 1.3: Sexism Categorization in Tweets
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

### Subtask 2.2: Source Intention in Memes
As in Subtask 1.2, this subtask aims to categorize memes according to the **author's intention**, providing insight into the role of social networks in disseminating sexist messages.  

Due to the characteristics of memes, the **REPORTED** label is virtually absent. Systems should classify memes only as **DIRECT** or **JUDGEMENTAL**.  
<img width="594" height="1235" alt="image" src="https://github.com/user-attachments/assets/b9ed9841-e0df-434f-9c99-067054d879a6" />

### Subtask 2.3: Sexism Categorization in Memes
This task classifies sexist memes using the same categories as Subtask 1.3:

1. **IDEOLOGICAL AND INEQUALITY**  
2. **STEREOTYPING AND DOMINANCE**  
3. **OBJECTIFICATION**  
4. **SEXUAL VIOLENCE**  
5. **MISOGYNY AND NON-SEXUAL VIOLENCE**

The following figures are some examples of categorized memes.

<img width="1063" height="924" alt="image" src="https://github.com/user-attachments/assets/45d6eabd-ef9e-48b5-b16b-bb0969e1ac06" />

## Research Objectives

- Compile and preprocess datasets containing social media posts, organized at the sentence and conversation level.
- Design and train ML models to:
  - Detect symptom-specific sentence relevance (Task 1).
  - Model user-level depression probability through temporal conversation analysis (Task 2).
- Rank sentences based on their alignment with the following 21 BDI-II symptoms:

  1. Sadness
  2. Pessimism
  3. Feelings of failure
  4. Loss of pleasure
  5. Feelings of guilt
  6. Feelings of punishment
  7. Dissatisfaction with oneself
  8. Severe self-criticism
  9. Suicidal thoughts or desires
  10. Crying
  11. Agitation
  12. Loss of interest
  13. Indecisiveness
  14. Loss of energy
  15. Changes in sleeping habits
  16. Irritability
  17. Changes in appetite
  18. Difficulty concentrating
  19. Tiredness or fatigue
  20. Loss of interest in sex
  21. Physical disorders related to emotional state

---

## Data Sets

The dataset used in this project was collected from TikTok using Apify’s TikTok Hashtag Scraper tool, focusing on hashtags potentially associated with sexist content. A rigorous manual selection process was conducted to ensure a balanced representation of both positive and negative seed hashtags. In total, 185 Spanish hashtags and 61 English hashtags were selected, ensuring a broad and representative collection of sexist-related content in both languages.

The collected videos were divided into training and test sets following a chronological and author-based partitioning strategy. This approach preserves temporal coherence while preventing data leakage: authors present in the training set were excluded from the test set, reducing the risk of the model learning author-specific patterns and improving generalization capabilities. Furthermore, each hashtag was required to contribute a minimum number of videos, ensuring a more uniform distribution across the dataset. Final video selection was performed randomly, while maintaining temporal distribution to avoid overrepresentation of any particular time period.

The final dataset consists of over 3,000 videos, with the training set containing 2,524 videos (1,524 Spanish and 1,000 English) and the test set containing 674 videos (304 Spanish and 370 English).




## Output

- **Task 1:**
  - A ranked list of depressive symptoms most relevant to each sentence analyzed.
- **Task 2:**
  - A prediction score indicating the likelihood of depression for each user based on sequential conversation modeling.
- **Task 3 (TBD):**
  - A prototype of a conversational agent capable of real-time mental health state assessment through dialogue.

---

## Requirements

- Strong motivation and interest in mental health applications and social media analysis.
- Good programming skills, especially in Python (experience with PyTorch or TensorFlow is advantageous).
- Familiarity with Natural Language Processing (NLP).



## Contact

_To be added_
