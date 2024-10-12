# CrackSegDiff: Diffusion Probability Model-based Multi-modal Crack Segmentation

## Abstract

Integrating grayscale and depth data in road inspection robots could enhance the accuracy, reliability, and comprehensiveness of road condition assessments, leading to improved maintenance strategies and safer infrastructure. However, these data sources are often compromised by significant background noise from the pavement. Recent advancements in Diffusion Probabilistic Models (DPM) have demonstrated remarkable success in image segmentation tasks, showcasing potent denoising capabilities, as evidenced in studies like SegDiff. Despite these advancements, current DPM-based segmentors do not fully capitalize on the potential of original image data. In this paper, we propose a novel DPM-based approach for crack segmentation, named CrackSegDiff, which uniquely fuses grayscale and range/depth images. This method enhances the reverse diffusion process by intensifying the interaction between local feature extraction via DPM and global feature extraction. Unlike traditional methods that utilize Transformers for global features, our approach employs Vm-unet to efficiently capture long-range information of the original data. The integration of features is further refined through two innovative modules: the Channel Fusion Module (CFM) and the Shallow Feature Compensation Module (SFCM). Our experimental evaluation on the three-class crack image segmentation tasks within the FIND dataset demonstrates that CrackSegDiff outperforms state-of-the-art methods, particularly excelling in the detection of shallow cracks.

Paper: [arxiv](https://arxiv.org/abs/2410.08100)

## A Quick Overview 

<div align="center">
  <img width=680 src="https://github.com/sky-visionX/CrackSegDiff/blob/CrackSegdiff/1%20.png">
  <p><em>The overall architecture of CrackSegDiff.</em></p>
</div>

## Quantization Results of CrackSegDiff

<div align="center">
  <img width=680 src="https://github.com/sky-visionX/CrackSegDiff/blob/CrackSegdiff/1%20.png">
  <p><em>The overall architecture of CrackSegDiff.</em></p>
</div>

## 1.Requirement

``pip install -r requirement.txt``

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy). After that, the pre-trained weights should be stored in './pretrained_weights/'.

## Example Cases
### pixel-level ground truth crack data for deep learning-based crack segmentation
1. Download FIND dataset from https://zenodo.org/records/6383044.
  
2. For training, run: ``python CrackSegDiff/segmentation_train.py --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 8``

3. For sampling, run: ``python CrackSegDiff/segmentation_sample.py  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 1``

4. For evaluation, run ``python CrackSegDiff/segmentation_env.py --inp_pth XXX  --out_pth XXX``


In default, the samples will be saved at `` ./results/`` 

## Thanks
Code copied a lot from [MedSegDiff](https://github.com/MedicineToken/MedSegDiff), [guided-diffusion](https://github.com/openai/guided-diffusion), [SegDiff](https://github.com/tomeramit/SegDiff), [VM-UNet](https://github.com/JCruan519/VM-UNet), 
## Cite
Please cite
~~~
@inproceedings{Jiang2024CrackSegDiffDP,
  title={CrackSegDiff: Diffusion Probability Model-based Multi-modal Crack Segmentation},
  author={Xiaoyan Jiang and Licheng Jiang and Anjie Wang and Kaiying Zhu and Yongbin Gao},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:273233791}
}
~~~
