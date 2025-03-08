import tkinter as tk
from tkinter import messagebox

def submit_form():

    text_input = text_entry.get("1.0", tk.END)
    
    # Validate input
    if not text_input.strip():
        messagebox.showerror("Error", "Text input is required!")
        return
    

    messagebox.showinfo("Success", f"Form submitted successfully!\nText: {text_input}")
    

    text_entry.delete("1.0", tk.END)


root = tk.Tk()
root.title("Text Input Form")
root.geometry("700x500")
root.configure(padx=20, pady=20)


form_frame = tk.Frame(root)
form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)


label = tk.Label(form_frame, font=("Arial", 12), text="deepseek-r1:1.5b:")
label.pack(anchor="center", pady=(0, 10))

text_entry = tk.Text(form_frame, font=("Arial", 12), width=30, height=10)
text_entry.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


# Create a loading indicator that will be shown during submission
loading_indicator = tk.Label(form_frame, text="Processing...", font=("Arial", 12), fg="#4CAF50")
loading_indicator.pack(pady=5)
loading_indicator.pack_forget()  # Hide initially

def submit_with_animation():
    # Show loading indicator
    submit_button.pack_forget()
    loading_indicator.pack(pady=5)
    root.update()
    
    # Simulate processing delay
    root.after(1000, lambda: [submit_form(), 
                             loading_indicator.pack_forget(),
                             submit_button.pack(pady=10)])

submit_button = tk.Button(form_frame, text="Submit", font=("Arial", 12, "bold"), 
                         bg="#4CAF50", fg="white", padx=20, pady=5,
                         command=submit_with_animation)
submit_button.pack(pady=10)


if __name__ == "__main__":
    root.mainloop()
