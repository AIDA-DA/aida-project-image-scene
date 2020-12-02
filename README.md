# Scene Classification for the Intel Scene Classification Challenge
We will build an image classifier, using the Intel Image Classificaion dataset. The dataset contains 25k images of the 6 categories. Our goal is to develop an algorithm which can distinguish the different image categories. In this approach we will use Tensorflow Keras together with pretained models provided by the framework.

### Results
<table>
    <thead>
        <tr>
            <th rowspan=2>Algorithm</th>
            <th rowspan=2>Image Size</th>
            <th rowspan=2>Training-Method</th>
            <th rowspan=2>Online-Augmentation</th>
            <th colspan=2>Validation</th>
        </tr>
        <tr>
            <th>Acc-Score</th>
            <th>Loss</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>ResNet-50</td>
        <td>224</td>
        <td>ImageNet FeatureExtraction - Only Dense Layer Training</td>
        <td>yes</td>
        <td>78,90%</td>
        <td>0,5927</td>
    </tr>
    <tr>
        <td>ResNet-50</td>
        <td>224</td>
        <td>ImageNet FineTuning - first layers</td>
        <td>yes</td>
        <td>93,50%</td>
        <td>0,2465</td>
    </tr>
    <tr>
        <td>ResNet-50</td>
        <td>224</td>
        <td>ImageNet - FineTuning - last layers</td>
        <td>yes</td>
        <td>93,87%</td>
        <td>0,2319</td>
    </tr>
    <tr>
        <td>ResNet-50</td>
        <td>224</td>
        <td>ImageNet FineTuning - all layers</td>
        <td>yes</td>
        <td>93,87%</td>
        <td>0,2411</td>
    </tr>
    <tr>
        <td>ResNet-50</td>
        <td>224</td>
        <td>Training from Scratch</td>
        <td>yes</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <td>MobileNetV2</td>
        <td>150</td>
        <td>ImageNet FineTuning - first layers</td>
        <td>yes</td>
        <td>92,47%</td>
        <td>0,2416</td>
    </tr>
    <tr>
        <td>Simple Conv Oliver</td>
        <td>150</td>
        <td>Training from Scratch</td>
        <td>no</td>
        <td>84,70%</td>
        <td>0,5192</td>
    </tr>
    <tr>
        <td>Simple Conv Oliver</td>
        <td>150</td>
        <td>Training from Scratch</td>
        <td>yes</td>
        <td>86,93%</td>
        <td>0,4044
    </tbody>
</table>
