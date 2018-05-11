## casUS_seg_ctx_re

## Fully-Automated Segmentation of Fetal Head in Prenatal Volumetric Ultrasound
## *Abstract:*  
Biometrics of fetal head are important indicators for maternal and fetal health monitoring during pregnancy. Volumetric ultrasound presents unique superiorities over traditional planar ultrasound scanning in encoding those biometrics and may promote the followed diagnoses. However, automatically differentiating the fetal head in volumetric ultrasound still pends as an emerging and unsolved problem. The challenges that automated segmentation solutions need to tackle include the poor image quality, inevitable boundary ambiguity, longspan occlusion, and the intrinsic appearance variability across different poses and gestational ages. In this paper, we propose the first fully-automated solution to segment fetal head in volumetric ultrasound. Receiving the whole ultrasound volume
as input, we formulate the segmentation task as an end-toend volumetric mapping by taking carefully tailored 3D fully convolutional networks (FCNs) as cores. Auxiliary supervisions with GPU-friendly design are injected to combat the curse of low training efficiency faced by 3D FCN. To enhance the local spatial consistency in the segmentation result, we further organize multiple 3D FCNs in a cascaded fashion to refine the prediction by iteratively revisiting the context represented by predictions from predecessors. Finally, we adopt a Random Erasing strategy to augment training corpus and thus make our models more robust against the ubiquitous boundary ambiguity and occlusion in ultrasound volumes. Extensively verified on large datasets, our method presents superior segmentation performance, high
agreements with experts and decent reproducibilities, and therefore is very promising to be a feasible solution in advancing the volumetric ultrasound based routine prenatal screening.

***
### TensorFlow 1.4

### Python 3.4

### [skimage](http://scikit-image.org/), [nibabel](http://nipy.org/nibabel/) need be installed

### stage_1 is USegNet-DS-RE Network,

### stage_2 is USegNet-Ctx-L1 Network，

if you want to run the stage_x,
please cd stage_x and set the parameter and the common is below
```
cd stage_x
python main.py
```
