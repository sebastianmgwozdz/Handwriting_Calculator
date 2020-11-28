import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from Model import *
import cv2
from ImageProcessor import *

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.model = create_model()
        self.create_widgets()

    def create_widgets(self):
        self.calc = tk.Button(self)
        self.calc["text"] = "Calculate"
        self.calc["command"] = self.extract_vals
        self.calc.pack()

        self.pick_file = tk.Button(self)
        self.pick_file["text"] = "Select File"
        self.pick_file["command"] = self.browse_file
        self.pick_file.pack()

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack()

    def display_text(self, text):
        self.text = tk.Label(self, text=text)
        self.text.pack()



    def show_image(self):
        self.img = ImageTk.PhotoImage(Image.open(self.img_path).resize((400, 400)))

        attributes = {"text", "incorrect", "correct", "e1", "adjust"}
        self.clear(attributes)


        if self.exists("img_panel"):
            self.img_panel.config(image=self.img)
            self.img_panel.photo_ref = self.img
        else:
            self.img_panel = tk.Label(self, image=self.img)
            self.img_panel.pack(side="bottom")




    def extract_vals(self):
        if not self.exists("img_path"):
            self.display_text("Must select an image first")
            return

        proc = ImageProcessor()

        self.pred_values = []
        self.images = []

        for x, y, w, h in proc.contour_boxes(self.img_path):

            img = prepare_img(cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w])
            self.pred_values.append(str(self.model.predicted_value(img)))
            self.images.append(img)

        string = ""
        for val in self.pred_values:
            string += val


        self.display_text(string + "\nIs this correct?")

        if not self.exists("incorrect"):
            self.incorrect = tk.Button(self)
            self.incorrect["text"] = "No"
            self.incorrect["command"] = self.fix
        self.incorrect.pack()

        if not self.exists("correct"):
            self.correct = tk.Button(self)
            self.correct["text"] = "Yes"
            self.correct["command"] = self.calculate
        self.correct.pack()


    def calculate(self):
        left = ""
        operand = ""
        operation = -1
        for val in self.pred_values:
            if LABELS[val] > 9:
                left = operand
                operand = ""
                operation = LABELS[val]
            else:
                operand += val

        self.display_text(left + " " + operation + " " + operand)

    def fix(self):
        attributes = {"text", "incorrect", "correct", "e1", "adjust"}
        self.clear(attributes)

        self.e1 = tk.Entry(self)
        self.e1.pack()

        self.adjust = tk.Button(self)
        self.adjust["text"] = "Train"
        self.adjust["command"] = self.refit
        self.adjust.pack()

    def refit(self):
        correct_expression = self.e1.get()
        ind = 0

        for i, val in enumerate(self.pred_values):
            corr_val = correct_expression[ind:ind + len(val)]
            if val != corr_val:
                self.model.compile()

                corr_val = LABELS[corr_val]

                self.model.train(self.images[i], np.reshape(corr_val, (1,)), epochs=100)
                self.extract_vals()

            ind += len(val)

        attributes = {"e1", "adjust"}
        self.clear(attributes)


    def browse_file(self):
        self.img_path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                   filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("jpeg", "*.jpeg"), ("png", "*.png")))
        self.show_image()

    def exists(self, attr):
        return hasattr(self, attr)

    def clear(self, attributes):
        for attr in attributes:
            if self.exists(attr):
                elem = getattr(self, attr)
                elem.pack_forget()



def prepare_img(img):
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (28, 28))
    img = np.reshape(img, (1, 28, 28))
    img = normalize(img)

    return img

def configure_window(win):
    win.geometry("1920x1080")

    # Centers the window
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()

def create_model():
    model = ConvolutionalNN()

    try:
        model.load("model.json", "model.h5")
    except OSError as e:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_symbols, y_symbols = load_images_and_labels('archive', {
            'minus': 10,
            'plus': 11,
            'times': 12
        }, (28, 28))

        x_symbols_train, x_symbols_test, y_symbols_train, y_symbols_test = train_test_split(
            x_symbols, y_symbols, test_size=0.2, random_state=1)

        x_train = combine_rows(x_train, x_symbols_train)

        x_train = normalize(x_train)

        y_train = combine_rows(y_train, y_symbols_train)


        model.compile()
        model.train(x_train, y_train, epochs=2)
        model.save('model.json', "model.h5")

    return model



root = tk.Tk()
configure_window(root)
app = Application(master=root)
app.mainloop()
