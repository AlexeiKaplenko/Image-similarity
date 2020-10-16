# Image-similarity

Custom metric for image similarity evaluation for sequences of images (video).
This metric is used as a signal for HW to initiate new EDS map collection (current image is too different from the last image for which EDS map was collected).

Siamese network with triplet loss was used.
Given anchor image, algorithm evaluates similarity score (in the range 0 and 1) between anchor image and any single frame in the sequence
