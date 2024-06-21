import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model=load_model('BrainTumor10Epochs.keras')

#model=load_model('BrainTumor10Epochscategoriacl.keras')
image=cv2.imread('C:\\Users\\NEHA KUMARI\\Downloads\\archive\\pred\\pred2.jpg')

#image = image.load_img('C:\\Users\\NEHA KUMARI\\Downloads\\archive\\pred\\pred0.jpg')
#input image 
#plt.imshow(image,interpolation='nearest')
#plt.show()

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

#print(img)
input_img=np.expand_dims(img,axis=0)
result = model.predict(input_img)
print(result)

#edit 
predicted_class = np.argmax(result)
#print("Predicted class:", predicted_class)

# If you want to print the class labels
class_labels = ['No Tumor', 'Tumor']
predicted_label = class_labels[predicted_class]
#print("Predicted label:", predicted_label)

# Display the image and predicted label
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot the image
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Input Image")
axes[0].axis('on')

# Display the predicted result
if result[0][0] == 1:
    mess_disp="Yes,Brain Tumor Detected. "
else:
    mess_disp="No,Brain Tumor Detected "


axes[1].text(0.5, 0.5, f"{mess_disp}", fontsize=12, ha='center', va='center')
#axes[1].axis('off')

#axes[1].imshow(result, cmap='viridis', aspect='auto')  # Adjust the colormap as needed
axes[1].set_title("Predicted Result")
axes[1].axis('off')

plt.show()

# Display the predicted result

