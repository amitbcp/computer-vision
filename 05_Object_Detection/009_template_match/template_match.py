import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('cutouts/bbox-example-image.jpg')
#image = mpimg.imread('temp-matching-example-2.jpg')
templist = [
  'cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg', 'cutout4.jpg', 'cutout5.jpg',
  'cutout6.jpg'
]


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
  # Make a copy of the image
  imcopy = np.copy(img)
  # Iterate through the bounding boxes
  for bbox in bboxes:
    # Draw a rectangle given bbox coordinates
    cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
  # Return the image copy with boxes drawn
  return imcopy


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
  # Define an empty list to take bbox coords
  bbox_list = []
  # Define matching method
  # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
  #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
  method = cv2.TM_CCOEFF_NORMED
  # Iterate through template list
  for temp in template_list:
    # Read in templates one by one
    tmp = mpimg.imread('cutouts/' + temp)
    # Use cv2.matchTemplate() to search the image
    result = cv2.matchTemplate(img, tmp, method)
    # Use cv2.minMaxLoc() to extract the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # Determine a bounding box for the match
    w, h = (tmp.shape[1], tmp.shape[0])
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
      top_left = min_loc
    else:
      top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # Append bbox position to list
    bbox_list.append((top_left, bottom_right))
    # Return the list of bounding boxes
  return bbox_list


print("Finding Matches ...")
bboxes = find_matches(image, templist)
print("Marking Rectangles ...")
result = draw_boxes(image, bboxes)
print("Display !")
plt.imshow(result)
plt.show()