# Thermal Comfort Classification
Pytorch based model for classification of thermal comfort states using a multi modal dataset.

## Data and Preprocessing
| feature  | dtype  |   range    | data after preprocessing |
| ---------| ------ | -----------| ------------- |	 			
| age 	   | float32|   [18, _]	 | [0.0, 1.0] 	 |			
| gender   | int8	|	[0, 2]	 | one-hot-encoding (classes=3) | 	
| pmv-index| float32|   [-3.0, 3.0] | no changes
|  rgb     | float32|   ([0, 255], [0, 255], [0, 255]) | ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
|keypoints | float32|   ([0, 1920], [0, 1080], [0, 5000]) | ([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]) 
|heart-rate| float32|	[_, _] | [0.0, 1.0] |		
|wrist-temp| float32| 	[_, _] | [0.0, 1.0]	|	
|ambient-temp| float32|	[18.4, 32.2] | [0.0, 1.0] |	
|relative-humidity| float32| [0.2 , 0.41]| [0.0, 1.0] |
|emotion| int8|	[0, 5] | one-hot-encoding (classes=6) |		
|Label| float32| [-3.0, 3.0] |	no changes |				    

Normalization parameters for normalized features:
* __age__: min=0, max=100
* __rgb__: min=0, max=255 (for all 3 channels)
* __keypoints__: min=0 max=1920 (x), min=0 max=1080 (y), min=0 max=5000 (z)
* __heart-rate__: min=35, max=100
* __wrist-temp__: min and max from data
* __ambient-temp__: min and max from data
* __relative-humidity__: data already in [0, 1]
