import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import ttkbootstrap as ttk

from train import MnistModel
import dataset

dset = dataset.Dataset("data\\train.csv", "data")
data = dset.getData()
mnistModel = MnistModel(*data)
model = mnistModel.getModel()
scaler = mnistModel.getSceler()

# تابع برای به‌روزرسانی تصویر دوربین در GUI
def update_frame():
    ret, frame = cap.read()
    if ret:
        # تبدیل تصویر به فرمت مناسب برای نمایش در tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_gray = cv2.resize(frame_gray, (28, 28))
        # frame_gray = frame_gray.reshape(-1)
        adapted_thresh = cv2.adaptiveThreshold(
            frame_gray,
            300, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21, 
            2
        )
        adapted_thresh = adapted_thresh.flatten().reshape(1, -1)
        adapted_thresh = scaler.transform(adapted_thresh)
        a = model.predict(adapted_thresh)
        print(a)

        img = Image.fromarray(frame)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)  # تغییر اینجا
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, update_frame)  # هر ۱۰ میلی‌ثانیه تصویر را به‌روز کن

# تابع برای نمایش متن در نوار پایین
def show_text():
    text = text_entry.get()
    status_label.config(text=text)

# ایجاد پنجره اصلی
root = ttk.Window(themename="cosmo")  # استفاده از تم متریال دیزاین
root.title("دوربین با نوار وضعیت")
root.geometry("680x600")

# ایجاد برچسب برای نمایش تصویر دوربین
camera_label = ttk.Label(root)
camera_label.pack(pady=10)

# ایجاد نوار وضعیت (پایین پنجره)
status_frame = ttk.Frame(root)
status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

# ایجاد ورودی متن
text_entry = ttk.Entry(status_frame, width=50)
text_entry.pack(side=tk.LEFT, padx=5)

# ایجاد دکمه برای نمایش متن
text_button = ttk.Button(status_frame, text="نمایش متن", command=show_text)
text_button.pack(side=tk.LEFT, padx=5)

# ایجاد برچسب برای نمایش متن وارد شده
status_label = ttk.Label(status_frame, text="متن اینجا نمایش داده می‌شود", font=("Arial", 12))
status_label.pack(side=tk.LEFT, padx=5)

# باز کردن دوربین
cap = cv2.VideoCapture(0)  # عدد ۰ نشان‌دهنده دوربین پیش‌فرض است
if not cap.isOpened():
    status_label.config(text="دوربین باز نشد!")
    raise ValueError("دوربین باز نشد!")

# شروع به‌روزرسانی تصویر دوربین
update_frame()

# اجرای پنجره
root.mainloop()

# آزاد کردن دوربین بعد از بسته شدن پنجره
cap.release()