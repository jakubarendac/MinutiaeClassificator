# MinutiaeClassificator

MinutiaeClassificator is a Python library for extracting and classifiying minutiae from fingerprint images.

MinutiaeClassificator contains 2 modules:

* **MinutiaeNet** - module responsible for extracting minutiae points from fingerprint image. Using neural network architecture from [MinutiaeNet](https://github.com/luannd/MinutiaeNet)
* **ClassifyNet** - module responsible for classifying extraced minutiae points. Architecture based on FineNet module of MinutiaeNet

## Requirements: software

* **Python 2.7** - we are planning to update module for Python 3.x in future
* **CUDA** - MinutiaeClassificator using TensorFlow GPU acceleration

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar. We reccomend to use it in **anaconda** enviroment. Installation in anaconda enviroment:

```bash
conda install cudatoolkit=<version compatible with the system CUDA>
pip install minutiaeclassificator
```

## API

### Import modules

* `MinutiaeClassificator.exceptions.MinutiaeClassificatorExceptions` - module containing library specific exceptions:

   * `CoarseNetPathMissingException`
   * `FineNetPathMissingException`
   * `ClassifyNetPathMissingException`
   * `MinutiaeNetNotLoadedException`
   * `ClassifyNetNotLoadedException`


* `MinutiaeClassificator.MinutiaeClassificatorWrapper` - main module contains library module `MinutiaeClassificator`. It is main module for accessing library. It contains methods:

   * `get_coarse_net_path(coarse_net_path)` - used for setting path to pretrained model of submodule CoarseNet.
   * `get_fine_net_path(fine_net_path)` - used for setting path to pretrained model of submodule FineNet.
   * `get_classify_net_path(classify_net_path)` - used for setting path to pretrained model of submodule ClassifyNet.
   * `load_extraction_module()` - used for compiling extraction module MinutiaeNet. Throws `CoarseNetPathMissingException` or `FineNetPathMissingException` when missing path to respective model's weights file
   * `load_classification_module()` - used for compiling classification module ClassifyNet. Throws `ClassifyNetPathMissingException` when missing path to its weights file
   * `get_extracted_minutiae(image_path, as_image = True)` - used for extracting minutiae points from input image. Image is determined by `image_path` (path to image file). When `as_image = True` minutiae points are marked in input image and updated image is returned as `PIL.Image`. If `as_image = False` minutiae_points are returned as `numpy.array`. If MinutiaeNet not loaded throws `MinutiaeNetNotLoadedException`
   * `get_classified_minutiae(image_path, extracted_minutiae, as_image = True)` - used for classifying  extracted minutiae points. Accepts same arguments as previous method and additionaly `extracted_minutiae`, which is `numpy.array` in same shape as `get_extracted_minutiae` output. If ClassifyNet not loaded throws `ClassifyNetNotLoadedException`
   * `get_extracted_and_classified_minutiae(image_path, as_image = True)` - wrapper over previous two methods. Extracts and then classify extracted minutiae.
   * `get_single_classified_minutiae(minutiae_patch_path)` - used for classification of singe minutiae point image. Image is determined by `minutiae_patch_path`(path to image file). In version **1.0.0**, library is able to classify minutiae points into 6 classes:
      * **ending**
      * **bifurcation**
      * **fragment**
      * **enclosure**
      * **crossbar**
      * **other**

### Models
- **CoarseNet**: [Googledrive](https://drive.google.com/file/d/1alvw_kAyY4sxdzAkGABQR7waux-rgJKm/view?usp=sharing)    ||    [Dropbox](https://www.dropbox.com/s/gppil4wybdjcihy/CoarseNet.h5?dl=0)
- **FineNet**: [Googledrive](https://drive.google.com/file/d/1wdGZKNNDAyN-fajjVKJoiyDtXAvl-4zq/view?usp=sharing)    ||    [Dropbox](https://www.dropbox.com/s/k7q2vs9255jf2dh/FineNet.h5?dl=0)
- **ClassifyNet**: [Googledrive](https://drive.google.com/drive/folders/124M3iLy4yMlAtegO0OXo_bl4Q0IIgPWE)
 

## Usage

```python
from MinutiaeClassificator.MinutiaeClassificatorWrapper import MinutiaeClassificator
from MinutiaeClassificator.exceptions.MinutiaeClassificatorExceptions import 
                                                        ClassifyNetPathMissingException

minutiaeClassificator = MinutiaeClassificator()
minutiaeClassificator.get_classify_net_path('path to file')

try:
    minutiaeClassificator.load_classification_module()
except ClassifyNetPathMissingException:
    do something...
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)