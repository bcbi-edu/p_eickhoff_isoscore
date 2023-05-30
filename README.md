# IsoScore & I-STAR 
This repository contains code for two related projects: IsoScore and I-STAR. In IsoScore, we develop the first tool capable of accurately measuring isotropy in embedding space. In I-STAR, we develop a novel regularization penalty based on an improved version of IsoScore, IsoScore⋆, that is capable of decreasing and increasing isotropy in embedding space. Full details of each project are contained in the directories IsoScore and I-STAR. To easily access IsoScore, IsoScore⋆ and use I-STAR loss for fine-tuning models, you can install IsoScore via PyPI as follows:

```
pip install IsoScore
```

## IsoScore: Measuring the Uniformity of Embedding Space Utilization 
The first paper proposes IsoScore: a novel tool that quantifies the degree to which a point cloud uniformly utilizes the ambient vector space. Using rigorously designed tests, we demonstrate that IsoScore is the only tool available in the literature that accurately measures how uniformly distributed variance is across dimensions in vector space. IsoScore was published in the Findings of the ACL 2022 (https://aclanthology.org/2022.findings-acl.262/. If you would like to cite this work, please refer to:

```bibtex
@inproceedings{rudman-etal-2022-isoscore,
    title = "{I}so{S}core: Measuring the Uniformity of Embedding Space Utilization",
    author = "Rudman, William  and
      Gillman, Nate  and
      Rayne, Taylor  and
      Eickhoff, Carsten",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.262",
    doi = "10.18653/v1/2022.findings-acl.262",
    pages = "3325--3339",
    }
```

## Stable Anisotropic Regularization
In this paper, we propose I-STAR: IsoScore⋆-based STable Anisotropic Regularization, a novel regularization method that can be used to increase or decrease levels of isotropy in embedding space during training. I-STAR uses IsoScore⋆, an improved version of IsoScore that is both differentiable and stable on mini-batch computations. In contrast to several previous works, we find that decreasing isotropy in LLMs tends to improve performance on a variety of fine-tuning tasks. I-STAR is currently under anonymous review, but the pre-print is available on arXiv (). 

