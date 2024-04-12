import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from PIL import Image
import torchvision.transforms as transforms
from model_definition import CNN
import numpy as np
from skimage import measure
import sympy as sp
import io
import base64
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = CNN(num_classes=21)  # Assuming 14 classes for equations
model.load_state_dict(torch.load('math_symbol_classifier_v2.pth', map_location=device))  # Load the model weights
model.eval()

# Define label mapping
LABEL_MAP = {
    'dot': '10',
    'minus': '11',
    'plus': '12',
    'slash': '13',
    'div': '14',
    'equal': '15',
    'times': '16',
    'w': '17',
    'x': '18',
    'y': '19',
    'z': '20' 
}

IDX_MAP = {int(idx): lbl for lbl, idx in LABEL_MAP.items()}


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

    # predicted_label = str(predicted.item())
    # predicted_class = LABEL_MAP.get(predicted_label, 'Unknown')
    predicted_label = predicted.item()
    
    # idx_map = {int(idx): lbl for lbl, idx in LABEL_MAP.items()}
    # Map the predicted label to the corresponding class
    if predicted_label < 10:
        predicted_class = predicted_label
    else:
        predicted_class = IDX_MAP[predicted_label]

    # Get confidence scores
    confidence_scores = torch.softmax(output, dim=1).squeeze().numpy()

    return str(predicted_class), confidence_scores

# Function to preprocess and provide top image classifcations
def get_top_symbol_classifications(image , k=4):
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
    
    # Get the probability scores for all classes
    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the top-K predictions
    _, top_indices = torch.topk(probabilities, k, dim=1)

    # Convert top-K indices to numpy arrays
    top_indices = top_indices.squeeze(0).numpy()
    
    # Get top-K predications and map predications to str symbols
    top_symbols = top_indices[:k]
    top_symbols = [IDX_MAP[item] if item in IDX_MAP else str(item) for item in top_symbols]
    
    return top_symbols
    
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
        'dot': '*',
        'minus': '-',
        'plus': '+',
        'slash': '/',
        'div': '\\div',
        'equal': '=',
        'times': '\times',
        'w': 'w',
        'x': 'x',
        'y': 'y',
        'z': 'z',
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
    try:
        # Convert LaTeX equation string to symbolic expression
        expr = sp.sympify(equation)

        print("----")
        print(expr)
        # Evaluate the expression to obtain the solution
        # solution = expr.evalf()
        return expr
    except (sp.SympifyError, TypeError, ValueError):
        return None
        

def main():
    st.title('Handwritten Equation Recognizer')
    
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Create sidebar for user input
    with st.sidebar:
        st.subheader("Canvas Settings")
        stroke_width = st.slider("Stroke Width", 1, 20, 10, key=1)
    
    # Create two columns
    col1, col2 = st.columns(2)

    
    with col1:
        # Create a canvas for drawing
        canvas = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#ffffff",
            height=200,
            width=200,
            drawing_mode="freedraw",
            key="canvas"
        )
        # Save button
        if st.button("Save Image"):
            # Get the drawn image from the canvas
            image_data = canvas.image_data.astype(np.uint8)
            uploaded_image = Image.fromarray(image_data)
            # uploaded_image = uploaded_image.convert("L")
            image_file = io.BytesIO()
            uploaded_image.save(image_file, format="PNG")
            st.session_state.uploaded_image = uploaded_image
        
    with col2:
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.session_state.uploaded_image = image
    
    # options = []
    # if st.session_state.uploaded_image is not None:
    #     options = get_top_symbol_classifications(st.session_state.uploaded_image)
    #     selected_option = st.radio("Specify the symbol you were trying to write:", options)
    #     print(selected_option)
        
    if st.session_state.uploaded_image is not None:
        # Recognize the equation
        recognized_equation = recognize_equation(st.session_state.uploaded_image)
        
        # Display the recognized LaTeX format equation
        st.subheader('Recognized Equation:')
        st.latex(recognized_equation)
        
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