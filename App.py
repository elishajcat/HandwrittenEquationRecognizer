import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_definition import CNN

# Load the pre-trained model
model = CNN(num_classes=14)  # Assuming 14 classes for equations
model.load_state_dict(torch.load('math_symbol_classifier.pth', map_location=torch.device('cpu')))  # Load the model weights
model.eval()

# Define label mapping
LABEL_MAP = {
    '10': 'dot',
    '11': 'minus',
    '12': 'plus',
    '13': 'slash'
}

def predict_equation(image):
    # Preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)

    _, predicted = torch.max(output, 1)
    predicted_label = str(predicted.item())
    predicted_class = LABEL_MAP.get(predicted_label, 'Unknown')

    return predicted_class

def main():
    st.title('Handwritten Equation Recognizer')

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Recognize the equation
        predicted_class = predict_equation(image)

        st.subheader('Prediction:')
        st.write(f"The predicted equation is: {predicted_class}")

if __name__ == '__main__':
    main()


# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import torch
# import torchvision.transforms as transforms
# from model_definition import CNN
# # import torch.nn as nn

# class CNN(nn.Module):
#     def __init__(self, num_classes):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Assuming input image size is 64x64
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = nn.functional.relu(self.conv3(x))
#         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
#         x = x.view(-1, 64 * 8 * 8)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Load the trained model
# model = CNN(num_classes=14)  # Assuming the same model architecture as in the training code
# model.load_state_dict(torch.load('math_symbol_classifier.pth'))
# # model.eval()

# # Define the label mapping
# LABEL_MAP = {
#     'dot': '10',
#     'minus': '11',
#     'plus': '12',
#     'slash': '13'
# }

# # Function to perform inference on the selected image
# def recognize_equation():
#     # Open file dialog to select an image
#     file_path = filedialog.askopenfilename()
#     if not file_path:
#         return

#     # Load and preprocess the image
#     image = Image.open(file_path).convert('L')
#     # Preprocess the image
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),  # Resize the image to match the input size expected by the model
#         transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#         transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
#     ])
#     image = transform(image)

#     # Add a batch dimension
#     image = image.unsqueeze(0)

#     # Set the model to evaluation mode
#     model.eval()

#     # Forward pass through the model
#     with torch.no_grad():
#         output = model(image)

#     # Get predicted label
#     _, predicted = torch.max(output, 1)

#     # Convert predicted label tensor to integer
#     predicted_label = predicted.item()

#     # Map the predicted label to the corresponding class
#     idx_map = {int(idx): lbl for lbl, idx in LABEL_MAP.items()}
#     predicted_class = idx_map[predicted_label]

#     # Show the result
#     messagebox.showinfo("Equation Recognition Result", f"The recognized symbol is: {predicted_class}")

# # Create the main application window
# root = tk.Tk()
# root.title("Handwritten Equation Recognition")

# # Create a button to select an image
# select_button = tk.Button(root, text="Select Image", command=recognize_equation)
# select_button.pack(pady=10)

# # Run the application
# root.mainloop()
