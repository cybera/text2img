<!-- For reference on dataset card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1 -->
<!-- Doc / guide: https://huggingface.co/docs/hub/datasets-cards -->

# Dataset Card for text2img

<!-- Provide a quick summary of the dataset. -->

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

- **Curated by:** [LAION.ai](https://laion.ai/)
- **Funded by [optional]:** N/A
- **Shared by [optional]:** Cybera, Inc
- **Language(s) (NLP):** English (American)
- **License:** Creative Common CC-BY 4.0

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/
- **Paper [optional]:** https://arxiv.org/abs/2210.08402
- **Demo [optional]:** https://huggingface.co/CompVis/stable-diffusion

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->
>Laion5B and LAION-400M could e.g. be used to train:\
\
    - Generative models: training image/text generative models, e.g autoregressive models like DALL-E or diffusion models like GLIDE \
    - Models with contrastive losses: self-supervised training on image/text pairs using contrastive losses, e.g CLIP \
    - Classification models: e.g, performing zero-shot classification by extracting pseudo labels from queries on the dataset
\
\
We present here a few examples of models that were trained on our LAION datasets with success:
(https://laion.ai/blog/laion-5b/#using-laion-datasets)

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

> The motivation behind dataset creation is to democratize research and experimentation around large-scale multi-modal model training and handling of uncurated, large-scale datasets crawled from publically available internet. Our recommendation is therefore to use the dataset for research purposes. Be aware that this large-scale dataset is uncurated. Keep in mind that the uncurated nature of the dataset means that collected links may lead to strongly discomforting and disturbing content for a human viewer. Therefore, please use the demo links with caution and at your own risk. It is possible to extract a “safe” subset by filtering out samples based on the safety tags (using a customized trained NSFW classifier that we built). While this strongly reduces the chance for encountering potentially harmful content when viewing, we cannot entirely exclude the possibility for harmful content being still present in safe mode, so that the warning holds also there. We think that providing the dataset openly to broad research and other interested communities will allow for transparent investigation of benefits that come along with training large-scale models as well as pitfalls and dangers that may stay unreported or unnoticed when working with closed large datasets that remain restricted to a small community. Providing our dataset openly, we however **do not recommend using it for creating ready-to-go industrial products, as the basic research about general properties and safety of such large-scale models, which we would like to encourage with this release, is still in progress**. - https://laion.ai/blog/laion-5b/#disclaimer-on-dataset-purpose-and-content-warning

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

>We provide these columns :\
\
    - URL: the image url, millions of domains are covered\
    - TEXT: captions, in english for en, other languages for multi and nolang\
    - WIDTH: picture width\
    - HEIGHT: picture height\
    - LANGUAGE: the language of the sample, only for laion2B-multi, computed using cld3\
    - similarity: cosine between text and image ViT-B/32 embeddings, clip for en, mclip for multi and nolang\
    - pwatermark: probability of being a watermarked image, computed using our watermark detector\
    - punsafe: probability of being an unsafe image, computed using our clip based detector\
    \
pwatermark and punsafe are available either as individual collections that must be joined with the hash of url+text, either as prejoined collections.
\- https://laion.ai/blog/laion-5b/#dataset-columns

## Dataset Creation

### Curation Rationale

>Since the release of CLIP & DALL-E in January 2021, several similar large multi-modal language-vision models have been trained by large groups. Models like FLORENCE, Turing Bletchley, ALIGN & BASIC demonstrated very strong transfer capabilities on novel datasets in absence of per-sample labels, which also steadily improved when growing training data amount, following scaling laws observed in previous research work. These models require billions of image-text pairs to achieve competitive performances and unfortunately, no billion-scale image-text pair dataset had been openly available up until now. To address this problem we release LAION 5B, a CLIP-filtered dataset of 5,85 billion high-quality image-text pairs, their CLIP ViT-L/14 embeddings, kNN-indices, a web interface for exploration & subset-creation and NSFW- and watermark-detection scores and tools. We describe the procedure to create the dataset and demonstrate successful training of DALL-E architecture. Having sufficiently large scales, the dataset opens venues for research on multi-modal language-vision models to a broad community.
\- https://laion.ai/blog/laion-5b/#introduction

### Source Data

> To create image-text pairs, we parse through WAT files from Common Crawl and parse out all HTML IMG tags containing an alt-text attribute. At the same time, we perform a language detection on text with three possible outputs: English language with confidence, another language with confidence, no language which contains “no detection” and “detection under the confidence threshold”. The “no language” set often contains short texts, mostly with names of people and places. All extracted information by the preprocessing workers were packed and sent to the Postgresql node for storage using the COPY command. The Postgresql server was maintained to keep about 500M records at all times by means of balancing the ingress and egress of data from the database.
\- https://laion.ai/blog/laion-5b/#distributed-processing-of-common-crawl

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

Refer to LAION's [Acquisition pipeline section](https://laion.ai/blog/laion-5b/#acquisition-pipeline)

#### Who are the source data producers?

LAION: Christoph Schuhmann, Richard Vencu, Romain Beaumont, Theo Coombes, Cade Gordon, Aarush Katta, Robert Kaczmarczyk, Jenia Jitsev

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

[More Information Needed]

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

Distributed filtering of Common Crawl using OpenAI CLIP (via [Crawling@Home](https://github.com/rvencu/crawlingathome-gpu-hcloud))

>To create image-text pairs, we parse through WAT files from Common Crawl and parse out all HTML IMG tags containing an alt-text attribute. At the same time, we perform a language detection on text with three possible outputs: English language with confidence, another language with confidence, no language which contains “no detection” and “detection under the confidence threshold”. The “no language” set often contains short texts, mostly with names of people and places. All extracted information by the preprocessing workers were packed and sent to the Postgresql node for storage using the COPY command. The Postgresql server was maintained to keep about 500M records at all times by means of balancing the ingress and egress of data from the database.  - https://laion.ai/blog/laion-5b/#distributed-processing-of-common-crawl

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

[More Information Needed]

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

Source data comes from Common Crawl, which is a freely available crawl of the public internet, and assumes data freely available online would not be considered sensitive.

For more information, refer to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use)

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

See [Out of Scope Use](#out-of-scope-use)

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

"Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations."

## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
```
@misc{schuhmann2022laion5b,
      title={LAION-5B: An open large-scale dataset for training next generation image-text models}, 
      author={Christoph Schuhmann and Romain Beaumont and Richard Vencu and Cade Gordon and Ross Wightman and Mehdi Cherti and Theo Coombes and Aarush Katta and Clayton Mullis and Mitchell Wortsman and Patrick Schramowski and Srivatsa Kundurthy and Katherine Crowson and Ludwig Schmidt and Robert Kaczmarczyk and Jenia Jitsev},
      year={2022},
      eprint={2210.08402},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

**APA:**

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., ... & Jitsev, J. (2022). Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35, 25278-25294.

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

[More Information Needed]

## More Information [optional]

It's important to note that in late 2023, a [Stanford study](https://www.forbes.com/sites/alexandralevine/2023/12/20/stable-diffusion-child-sexual-abuse-material-stanford-internet-observatory/?sh=390751585f21) found that some of the images contained within the LAION 5B dataset may be considered [CSAM](https://www.cybertip.ca/en/child-sexual-abuse/material/).

The LAION 5B dataset in question was used to train Stable Diffusion v1.5 and later, so _to the best of our knowledge_ there is no specific concern about the LAION 2B dataset used to train this model, and no reason to expect similar findings. If similar information comes to light about the LAION 2B dataset, this repo will either be updated to use a different model and dataset, or removed entirely.

## Dataset Card Authors [optional]

Jordan Swanson (Cybera, Inc); [More Information Needed]

## Dataset Card Contact

[More Information Needed]