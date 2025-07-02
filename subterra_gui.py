import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
try:
    model=joblib.load("subterra_decision_tree.pkl")
except FileNotFoundError:
    raise FaileNotFoundError("Model file 'suterra_decision_tree.pkl' not found in current directory.")


#Define the feature names (based on training data)
feature_names=[
    "heat_flow", "fault_distance", "dem", "tri", "ndvi",
    "lst", "slope", "vegetation_peak", "landcover", "lithology"
]

#Initialize te GUI window
root=tk.Tk()
root.title("SubTerra Geothermal Potential Classifier")
root.geometry("400x500")
root.resizable(False, False)

#STore entry widgets
entries={}

#Create input fields for each features
for idx, feature in enumerate(feature_names):
    tk.Label(root, text=f"{feature.replace('_', ' ').title()}:", anchor="w").grid(
        row=idx, column=0, padx=10, pady=5, sticky="e"
    )
    entry=tk.Entry(root, width=20)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[feature]=entry

#Function to classify input values
def classify():
    try:
        #Collect and validate input values
        values=[float(entries[feature].get()) for feature in feature_names]
        input_array=np.array([values])

        #Make predictions
        prediction=model.predict(input_array)[0]
        prob=model.predict_proba(input_array)[0][prediction]

        #Display result
        label="HIGH GEOTHERMAL POTENTIAL" if prediction==1 else "LOW GEOTHERMAL POTENTIAL"
        messagebox.showinfo("Prediction Result", f"{label}\nConfidence: {prob:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter numeric values only for all features")

#Add Classift button
tk.Button(
    root, text="Classify", command=classify, bg="#2E8B57", fg="white", font=("Arial", 12, "bold")
).grid(row=len(feature_names), column=0, columnspan=2, pady=20)

#Start GUI Loop
root.mainloop()