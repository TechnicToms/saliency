# saliency
A github repository containing various algorithms that generate saliency maps from images (Itti's Method ...). 


---

## Itti’s Method

The class “Ittis method” is the implementation of the paper: **A Model of Saliency-Based Visual Attention for Rapid Scene Analysis** (Short Paper: [1]).
Here, the class implements the adjacent image in the form of summarized functions.

![image](images/ittis_method.png)

Furthermore, the class is structured so that a simple and expert operation is possible. For the easy way of operation, the class can be simply imported and the `saliency_map` function can be called. As an expert, however, any subfunction of the class can be called and the intermediate results can be viewed. The `saliency_map` function therefore only summarizes the other subfunctions and calculates the overall result.
An easy example of the code is below:

```python
import torchvision
from ittis_method import IttisMethod

# Load rgb image
img = torchvision.io.read_image("data/cat.png")[0:3, ...] / 255.0

# Construct class and generate Saliency maps using Itti's method
ef = IttisMethod()
sm = ef.saliency_map(img)
```

TODO

**NOTE**: Because of the Pyramid images the in input image will be resized. As a limitation this class only takes images, that are larger than 256 px in height and width.


### _class_  IttisMethod(c: list = [2, 3, 4], delta: list = [3, 4])
A class for generating a saliency map using Itti’s method from the Paper:
`A Model of Saliency-Based Visual Attention for Rapid Scene Analysis`.


#### center_surrounded_differences(PyramidList_c: list, PyramidList_s)
Calculates the center-surrounded difference ($\ominus$) for the given pyramid images.

For a given pyramid image $I(\sigma)$ it performs the following equation for the two scales `c` and `s`:

I(s,c) = | I_c(c) \\ominus I_s(s) |


* **Parameters**

    
    * **PyramidList_c** (*list*) – list of pyramid images for c scales [from the previous function `linear_filtering(...)`]


    * **PyramidList_s** (*list*) – list of pyramid images for s scales [from the previous function `linear_filtering(...)`]



* **Raises**

    
    * **ValueError** – If length of input list PyramidList_c isn’t long enough


    * **ValueError** – If length of input list PyramidList_s isn’t long enough



* **Returns**

    csd maps



* **Return type**

    list



#### colors(img: Tensor)
Generates the color features for the given rgb image.


* **Parameters**

    **img** (*torch.Tensor*) – rgb input image



* **Returns**

    (R, G, B, Y) features



* **Return type**

    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]



#### intensity(img: Tensor)
Creates an intensity map from the given rgb image.


* **Parameters**

    **img** (*torch.Tensor*) – rgb input image



* **Returns**

    intensity map



* **Return type**

    torch.Tensor



#### linear_combination(I_dash: Tensor, C_dash: Tensor, O_dash: Tensor)
Calculates the saliency map S by the following formula:

S = \\frac{1}{3}\\left(\\mathcal{N}(\\bar{I}) + \\mathcal{N}(\\bar{C}) + \\mathcal{N}(\\bar{O}) \\right)


* **Parameters**

    
    * **I_dash** (*torch.Tensor*) – across-scale combinations of the intensity maps


    * **C_dash** (*torch.Tensor*) – across-scale combinations of the color maps


    * **O_dash** (*torch.Tensor*) – across-scale combinations of the orientation maps



* **Raises**

    **TypeError** – If one of the inputs in not of type torch.Tensor



* **Returns**

    saliency map S



* **Return type**

    torch.Tensor



#### linear_filtering(img: Tensor)
Performs the feature extraction and the follow up pyramid image generation.


* **Parameters**

    **img** (*torch.Tensor*) – input rgb image



* **Returns**

    Results of features



* **Return type**

    tuple



#### normalize_map(input_map: Tensor)
Normalizes the input map by the factor $(M-\mu)^2$,
where $M$ is the maximum of the map and $\mu$ the mean of the map.

\\mathcal{N}(\\mathbf{I}(x, y)) = (M-\\mu)^2 \\cdot \\mathbf{I}(x, y)


* **Parameters**

    **input_map** (*torch.Tensor*) – input map



* **Raises**

    
    * **TypeError** – If input is not of type torch.Tensor


    * **USerWarning** – If input map has more than one color channel



* **Returns**

    normalized map



* **Return type**

    torch.Tensor



#### orientations(img: Tensor)
Generates Gabor pyramids for the given input image.
For that purpose the image will be convertet using the `intensity` function and
then applied to the GaborPyramids class, which yield the gabor pyramids.


* **Parameters**

    **img** (*torch.Tensor*) – rgb input image



* **Returns**

    a list of orientations of pyramids



* **Return type**

    list



#### saliency_map(img: Tensor)
Calculates the Saliency map using the Itti’s method.

For that purpose the subfunctions of this class will be used, to first calculate the features, then the center-surround differences,
which are followed by the across-scale combination and in the end the linear combination of the three maps.

Everything that this function calculates can be also done manually, if the corresponding subfunctions of this class are called in the right order.

### Example

```python
>>> import torchvision
>>> img = torchvision.io.read_image("data/cat.png")[0:3, ...] / 255.0
>>> ef = IttisMethod()
>>> sm = ef.saliency_map(img)
```


* **Parameters**

    **img** (*torch.Tensor*) – input rgb image



* **Returns**

    saliency map



* **Return type**

    torch.Tensor
