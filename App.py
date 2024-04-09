import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_definition import CNN
import numpy as np
from skimage import measure
import sympy as sp

# Load the pre-trained model
model = CNN(num_classes=14)  # Assuming 14 classes for equations
model.load_state_dict(torch.load('math_symbol_classifier.pth', map_location=torch.device('cpu')))  # Load the model weights
model.eval()

# Define label mapping
LABEL_MAP = {
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': 'dot',
    '11': 'minus',
    '12': 'plus',
    '13': 'slash'
}

# Function to preprocess and classify an image
def classify_image(image):
    # Preprocess the input image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    _, predicted = torch.max(output, 1)
    predicted_label = str(predicted.item())
    predicted_class = LABEL_MAP.get(predicted_label, 'Unknown')

    # Get confidence scores
    confidence_scores = torch.softmax(output, dim=1).squeeze().numpy()

    return predicted_class, confidence_scores

# Function to recognize the handwritten equation
def recognize_equation(image):
    # Convert the image to grayscale
    image = image.convert('L')  # 'L' mode for grayscale

    # Thresholding to get binary image
    binary_image = np.array(image) < 200

    # Perform connected component analysis to extract symbols
    labels, num_labels = measure.label(binary_image, background=0, return_num=True)
    symbols = []
    bounding_boxes = []

    # Iterate through connected components and classify each symbol
    for label in range(1, num_labels + 1):
        # Extract the bounding box of the symbol
        indices = np.argwhere(labels == label)
        min_x, min_y = indices.min(axis=0)
        max_x, max_y = indices.max(axis=0)
        bounding_boxes.append((min_x, min_y, max_x, max_y))

        # Crop the symbol from the original image
        symbol_image = image.crop((min_y, min_x, max_y, max_x))

        # Classify the symbol
        symbol_class, _ = classify_image(symbol_image)
        symbols.append(symbol_class)

    # Sort symbols by their leftmost x-coordinate (min_y) to order them left to right
    sorted_indices = np.argsort([box[1] for box in bounding_boxes])
    symbols_sorted = [symbols[i] for i in sorted_indices]

    # Parse the recognized symbols into an equation string
    equation = make_equation(symbols_sorted)

    return equation

# Function to convert recognized symbols into LaTeX format equation
def make_equation(symbols):
    # Map symbols to LaTeX equivalents
    latex_mapping = {
        'dot': '\\cdot',
        'minus': '-',
        'plus': '+',
        'slash': '/',
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9'
    }

    # Convert symbols to LaTeX format
    latex_symbols = [latex_mapping[symbol] for symbol in symbols]

    # Join symbols into a LaTeX equation string
    latex_equation = ' '.join(latex_symbols)

    return latex_equation

# Function to solve the LaTeX format equation
def solve_equation(equation):
    # Convert LaTeX equation string to symbolic expression
    expr = sp.sympify(equation)
    
    print("----")
    print(expr)

    # Evaluate the expression to obtain the solution
    # solution = expr.evalf()

    return expr

def main():
    st.title('Handwritten Equation Recognizer')

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Recognize the equation
        recognized_equation = recognize_equation(image)

        # Display the recognized LaTeX format equation
        # st.subheader('Recognized Equation:')
        # st.latex(recognized_equation)
        
        # Solve the equation
        solution = solve_equation(recognized_equation)

        # Display the solution
        if solution:
            st.subheader('Equation Solution:')
            st.write(f"{recognized_equation} = {solution}")
        else:
            st.subheader('Equations Solution:')
            st.write("Could not reqcognize or solve the equation")

if __name__ == '__main__':
    main()







# import streamlit as st
# import torch
# from PIL import Image, ImageDraw
# import torchvision.transforms as transforms
# from model_definition import CNN

# # Load the pre-trained model
# model = CNN(num_classes=14)  # Assuming 14 classes for equations
# model.load_state_dict(torch.load('math_symbol_classifier.pth', map_location=torch.device('cpu')))  # Load the model weights
# model.eval()

# # Define label mapping
# LABEL_MAP = {
#     '10': 'dot',
#     '11': 'minus',
#     '12': 'plus',
#     '13': 'slash'
# }

# # Function to preprocess and classify an image
# def classify_image(image):
#     # Preprocess the input image
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
#     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

#     # Perform inference
#     with torch.no_grad():
#         output = model(image_tensor)

#     _, predicted = torch.max(output, 1)
#     predicted_label = str(predicted.item())
#     predicted_class = LABEL_MAP.get(predicted_label, 'Unknown')

#     return predicted_class

# # Function to recognize the handwritten equation
# def recognize_equation(image):
#     # Convert the image to grayscale
#     image = image.convert('L')  # 'L' mode for grayscale

#     # Perform segmentation and classification
#     symbols = []
#     digit_buffer = []

#     draw = ImageDraw.Draw(image)

#     width, height = image.size
#     for x in range(width):
#         for y in range(height):
#             pixel = image.getpixel((x, y))
#             if pixel < 200:  # Threshold for detecting ink (adjust as needed)
#                 draw.point((x, y), fill=0)  # Set the pixel to black for visualization
#                 digit_buffer.append((x, y))
#             else:
#                 if digit_buffer:
#                     # Extract the bounding box of the symbol
#                     min_x = min(coord[0] for coord in digit_buffer)
#                     max_x = max(coord[0] for coord in digit_buffer)
#                     min_y = min(coord[1] for coord in digit_buffer)
#                     max_y = max(coord[1] for coord in digit_buffer)

#                     # Crop the symbol from the original image
#                     symbol_image = image.crop((min_x, min_y, max_x, max_y))

#                     # Classify the symbol
#                     symbol_class = classify_image(symbol_image)
#                     symbols.append(symbol_class)

#                     digit_buffer = []

#     if digit_buffer:
#         # Process the last buffered digit
#         min_x = min(coord[0] for coord in digit_buffer)
#         max_x = max(coord[0] for coord in digit_buffer)
#         min_y = min(coord[1] for coord in digit_buffer)
#         max_y = max(coord[1] for coord in digit_buffer)

#         # Crop the symbol from the original image
#         symbol_image = image.crop((min_x, min_y, max_x, max_y))

#         # Classify the symbol
#         symbol_class = classify_image(symbol_image)
#         symbols.append(symbol_class)

#     # Parse the recognized symbols into an equation string
#     equation = ''.join(symbols)
#     return equation

# def main():
#     st.title('Handwritten Equation Recognizer')

#     uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#     if uploaded_image is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_image)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Recognize the equation
#         recognized_equation = recognize_equation(image)

#         st.subheader('Recognized Equation:')
#         st.write(f"The recognized equation is: {recognized_equation}")

# if __name__ == '__main__':
#     main()



# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk
# import torch
# import torchvision.transforms as transforms
# from model_definition import CNN
# # import torch.nn as nn

# # class CNN(nn.Module):
# #     def __init__(self, num_classes):
# #         super(CNN, self).__init__()
# #         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
# #         self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
# #         self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
# #         self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Assuming input image size is 64x64
# #         self.fc2 = nn.Linear(512, num_classes)

# #     def forward(self, x):
# #         x = nn.functional.relu(self.conv1(x))
# #         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
# #         x = nn.functional.relu(self.conv2(x))
# #         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
# #         x = nn.functional.relu(self.conv3(x))
# #         x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
# #         x = x.view(-1, 64 * 8 * 8)
# #         x = nn.functional.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

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
