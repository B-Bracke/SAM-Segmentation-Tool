"""
This script provides a GUI tool for image segmentation using the SAM model. The tool allows the user to load an image, select a prompt mode (single point or bounding box), and then select a point or draw a bounding box on the image to generate a segmentation mask. The mask can be saved as an image file. The tool uses the SAM model to generate the segmentation mask based on the user input.
Tool created by: Benjamin Bracke
Date: 2024-07-24
"""

import tkinter as tk
from tkinter import filedialog, StringVar
from PIL import Image, ImageTk
import copy
import torch
from transformers import SamModel, SamProcessor
import numpy as np
import os

class ImageSegmentationTool:
    def __init__(self, SAM_MODEL):
        self.SAM_MODEL = SAM_MODEL
        
        self.IMAGE = None
        self.IMAGE_VIEW = None
        self.IMAGE_EMBEDDING = None
        self.IMAGE_FILE = None
        self.MASK = None
        self.AVAILABLE_PROMPT_MODES = {"single point": 0, "bounding box": 1}
        self.MODE = 0

        # Initialize main window
        self.root = tk.Tk()
        self.root.title("SAM Image Segmentation Tool")
        # make it not resizable
        self.root.resizable(False, False)
        self.root.geometry("800x600")
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Add Image load/save buttons
        tk.Button(self.button_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT)
        tk.Label(self.button_frame, text="").pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Save Mask", command=self.save_mask, state=tk.DISABLED).pack(side=tk.LEFT)
        tk.Label(self.button_frame, text="").pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Save Cutout", command=self.save_cutout, state=tk.DISABLED).pack(side=tk.LEFT)

        # add radio buttons for switching between to prompt modes "single point" and "bounding box"
        self.radio_var = StringVar()
        self.radio_var.set("single point")
        tk.Label(self.button_frame, text="").pack(side=tk.LEFT)
        tk.Label(self.button_frame, text="Prompt Selection:").pack(side=tk.LEFT)
        tk.Radiobutton(self.button_frame, text="Single Point", variable=self.radio_var, value="single point",
                        command=self.on_radio_change).pack(side=tk.LEFT)
        tk.Radiobutton(self.button_frame, text="Bounding Box", variable=self.radio_var, value="bounding box",
                        command=self.on_radio_change).pack(side=tk.LEFT)
        tk.Label(self.button_frame, text="").pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Clear Prompts", command=self.clear_prompts, state=tk.DISABLED).pack(side=tk.LEFT)

        # Image display panel
        self.panel = tk.Canvas(self.root)
        self.panel.pack(side=tk.BOTTOM, expand=True)
        self.panel.bind("<Button-1>", self.point_event)

    def open_image(self):
        self.IMAGE_FILE = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.IMAGE_FILE:
            self.IMAGE = Image.open(self.IMAGE_FILE).convert("RGB")
            self.view_image(self.IMAGE)
            self.get_image_embedding()

    def view_image(self, image):
        self.IMAGE_VIEW = copy.deepcopy(image)
        self.IMAGE_VIEW.thumbnail((800, 600))
        # show the image on the panel
        self.panel.delete(tk.ALL)
        self.panel.image = ImageTk.PhotoImage(self.IMAGE_VIEW)
        self.panel.create_image(0, 0, image=self.panel.image, anchor=tk.NW)
        # change size of the panel to fit the image
        self.panel.config(width=self.IMAGE_VIEW.size[0], height=self.IMAGE_VIEW.size[1])

    def get_image_embedding(self):
        self.IMAGE_EMBEDDING = self.SAM_MODEL.get_embeddings(self.IMAGE)

    def extrapolate_coordinates(self, x, y):
        # Calculate the scaling factor
        scale_x = self.IMAGE.size[0] / self.IMAGE_VIEW.size[0]
        scale_y = self.IMAGE.size[1] / self.IMAGE_VIEW.size[1]
        # Calculate the extrapolated coordinates
        x = int(x * scale_x)
        y = int(y * scale_y)
        return x, y
    
    def show_mask_on_image(self, mask):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        mask = np.expand_dims(mask, axis=-1) * color
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask)
        img = copy.deepcopy(self.IMAGE).convert("RGBA")
        # alpha blending the mask with the image
        img.paste(mask, (0, 0), mask)
        self.view_image(img)

    
    def point_event(self, event):
        self.clear_prompts()
        # get the coordinates of the point and extrapolate them to the original image size and predict the mask from the point and show it on the image
        x,y = event.x, event.y
        x,y = self.extrapolate_coordinates(x, y)
        mask, _ = self.SAM_MODEL.predict_from_points(self.IMAGE, self.IMAGE_EMBEDDING, [[[x, y]]])
        self.MASK = mask
        self.show_mask_on_image(mask)
        self.prompted()

    def box_event(self, event):
        self.clear_prompts()
        # Start the bounding box selection
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.panel.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
        
        # Bind the methods to the panel
        self.panel.bind("<B1-Motion>", self.on_bbox_drag)
        self.panel.bind("<ButtonRelease-1>", self.on_bbox_release)

    def on_bbox_drag(self, event):
        # Update the bounding box selection as the mouse is dragged
        self.panel.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_bbox_release(self, event):
        # Finish the bounding box selection and get the coordinates of the bounding box
        x1, y1, x2, y2 = self.panel.coords(self.rect)
        self.panel.delete(tk.ALL)
        self.panel.unbind("<B1-Motion>")
        self.panel.unbind("<ButtonRelease-1>")

        # predict the mask from it
        x1, y1 = self.extrapolate_coordinates(x1, y1)
        x2, y2 = self.extrapolate_coordinates(x2, y2)
        mask, _ = self.SAM_MODEL.predict_from_boxes(self.IMAGE, self.IMAGE_EMBEDDING, [[[x1, y1, x2, y2]]])
        self.MASK = mask
        
        self.show_mask_on_image(mask)
        self.prompted()


    def prompted(self):
        # activate the save button and clear prompts button
        self.button_frame.winfo_children()[2].config(state=tk.NORMAL) # save mask button
        self.button_frame.winfo_children()[4].config(state=tk.NORMAL) # save cutout button
        self.button_frame.winfo_children()[10].config(state=tk.NORMAL) # clear prompts button

    def clear_prompts(self):
        self.MASK = None
        self.view_image(self.IMAGE)
        # deactivate the save button and clear prompts button
        self.button_frame.winfo_children()[2].config(state=tk.DISABLED) # save mask button
        self.button_frame.winfo_children()[4].config(state=tk.DISABLED) # save cutout button
        self.button_frame.winfo_children()[10].config(state=tk.DISABLED) # clear prompts button


    def save_mask(self):
        filename, extention = os.path.splitext(os.path.basename(self.IMAGE_FILE))
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=filename+"_mask.png",
                                                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")])
        if file_path:
            mask = Image.fromarray(self.MASK)
            mask.save(file_path)

    def save_cutout(self):
        filename, extention = os.path.splitext(os.path.basename(self.IMAGE_FILE))
        file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=filename+"_cutout.png",
                                                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")])
        if file_path:
            cutout = Image.fromarray(np.expand_dims(self.MASK, axis=2) * np.array(self.IMAGE))
            cutout.save(file_path)

    def on_radio_change(self):
        self.clear_prompts()
        self.MODE = self.AVAILABLE_PROMPT_MODES[self.radio_var.get()]
        if self.MODE == 1:
            self.panel.unbind("<Button-1>")
            self.panel.bind("<Button-1>", self.box_event)           
        else:
            self.panel.unbind("<Button-1>")
            self.panel.bind("<Button-1>", self.point_event)

    def run(self):
        # Run the application
        self.root.mainloop()





class SAMModel():
    def __init__(self, backbone="facebook/sam-vit-huge"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Using backbone: {backbone}")
        print("Loading model. This might take a while...")
        self.model = SamModel.from_pretrained(backbone).to(self.device)
        self.processor = SamProcessor.from_pretrained(backbone)

    def get_embeddings(self, image):
        print("Calculating image embeddings...")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
        return image_embeddings
    
    def predict_from_points(self, image, image_embeddings, input_points):
        inputs = self.processor(image, input_points=input_points, return_tensors="pt").to(self.device)
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        return self.forward_model(inputs)
    
    def predict_from_boxes(self, image, image_embeddings, input_boxes):
        inputs = self.processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        # pop the pixel_values as they are not neded
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        return self.forward_model(inputs)

    @torch.no_grad()
    def forward_model(self, inputs):
        outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        scores = outputs.iou_scores
        masks = masks.squeeze(dim=[0,1]).detach().cpu().numpy()
        scores = scores.squeeze(dim=[0,1]).detach().cpu().numpy()
        return masks, scores



# Create an instance of the ImageSegmentationTool class and run the application
sam_model = SAMModel()
segmentation_tool = ImageSegmentationTool(sam_model)
segmentation_tool.run()