# Listwise-View-Ranking-Image-Cropping

### Material
[Paper](https://arxiv.org/pdf/1905.05352.pdf), [Model]()

### Citation
```
@article{lu2019listwise,
  title={Listwise View Ranking for Image Cropping},
  author={Lu, Weirui and Xing, Xiaofen and Cai, Bolun and Xu, Xiangmin},
  journal={arXiv preprint arXiv:1905.05352},
  year={2019}
}
```

### Abstract
Rank-based Learning with deep neural network has been widely used for image cropping. However, the performance of ranking-based methods is often poor and this is mainly due to two reasons: 1) image cropping is a listwise ranking task rather than pairwise comparison; 2) the rescaling caused by pooling layer and the deformation in view generation damage the performance of composition learning. In this paper, we develop a novel model to overcome these problems. To address the first problem, we formulate the image cropping as a listwise ranking problem to find the best view composition. For the second problem, a refined view sampling (called RoIRefine) is proposed to extract refined feature maps for candidate view generation. Given a series of candidate views, the proposed model learns the Top-1 probability distribution of views and picks up the best one. By integrating refined sampling and listwise ranking, the proposed network called LVRN achieves the state-of-the-art performance both in accuracy and speed.

### How to run the Code
1. Download the cropping dataset(FCDB and FLMS) into ``data`` folder.
2. Download the pre-trained model into ``model`` folder. 
3. ``cd ./roi_crop `` and run ``make.sh`` to build roi_crop module.
4. modify the path of ``create_gt_crops.py`` and run ``python create_gt_crops.py`` to create the ground-truth.
5. run ``python demo.py --GPU x``

### Qualitative visualization on FCDB dataset
1st and 4th : GT 

2nd, 3rd, 5th, 6th : ours 
<figure class="third">
<img style="width:160px;height:90px"  src="LVRN_eval/images/0_gt.jpg"     alt="gt" align=center />
<img style="width:160px;height:90px"  src="LVRN_eval/images/0_our.jpg"    alt="our" align=center />
<img style="width:160px;height:70px"  src="LVRN_eval/images/0_crop.jpg"   alt="crop" align=center />

<img style="width:160px;height:120px" src="LVRN_eval/images/38_gt.jpg"    alt="gt" align=center />
<img style="width:160px;height:120px" src="LVRN_eval/images/38_our.jpg"   alt="our" align=center />
<img style="width:160px;height:90px"  src="LVRN_eval/images/38_crop.jpg"  alt="crop" align=center />
</figure>

<figure class="third">
<img style="width:160px;height:120px" src="LVRN_eval/images/28_gt.jpg"    alt="gt" align=center />
<img style="width:160px;height:120px" src="LVRN_eval/images/28_our.jpg"   alt="our" align=center />
<img style="width:160px;height:90px"  src="LVRN_eval/images/28_crop.jpg"  alt="crop" align=center />

<img style="width:160px;height:109px" src="LVRN_eval/images/168_gt.jpg"   alt="gt" align=center />
<img style="width:160px;height:109px" src="LVRN_eval/images/168_our.jpg"  alt="our" align=center />
<img style="width:150px;height:56px"  src="LVRN_eval/images/168_crop.jpg" alt="crop" align=center />
</figure>

<figure class="third">
<img style="width:160px;height:107px" src="LVRN_eval/images/29_gt.jpg"    alt="gt" align=center />
<img style="width:160px;height:107px" src="LVRN_eval/images/29_our.jpg"   alt="our" align=center />
<img style="width:160px;height:75px"  src="LVRN_eval/images/29_crop.jpg"  alt="crop" align=center />

<img style="width:160px;height:106px" src="LVRN_eval/images/183_gt.jpg"   alt="gt" align=center />
<img style="width:160px;height:106px" src="LVRN_eval/images/183_our.jpg"  alt="our" align=center />
<img style="width:115px;height:97px"  src="LVRN_eval/images/183_crop.jpg" alt="crop" align=center />
</figure>
