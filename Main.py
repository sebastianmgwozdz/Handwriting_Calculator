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

        attributes = {"text", "incorrect_btn", "correct_btn", "e1", "adjust"}
        clear(self, attributes)


        if hasattr(self, "img_panel"):
            self.img_panel.config(image=self.img)
            self.img_panel.photo_ref = self.img
        else:
            self.img_panel = tk.Label(self, image=self.img)
            self.img_panel.pack(side="bottom")

    def extract_vals(self):
        if not hasattr(self, "img_path"):
            self.display_text("Must select an image first")
            return

        proc = ImageProcessor()

        self.pred_values = []
        self.images = []

        for x, y, w, h in proc.contour_boxes(self.img_path):

            img = prepare_img(cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w])
            self.pred_values.append(self.model.predicted_value(img))
            self.images.append(img)

        string = ""
        for val in self.pred_values:
            string += LABELS[val]


        self.display_text(string + "\nIs this correct?")

        if not hasattr(self, "incorrect_btn"):
            self.incorrect_btn = tk.Button(self)
            self.incorrect_btn["text"] = "No"
            self.incorrect_btn["command"] = self.incorrect
        self.incorrect_btn.pack()

        if not hasattr(self, "correct_btn"):
            self.correct_btn = tk.Button(self)
            self.correct_btn["text"] = "Yes"
            self.correct_btn["command"] = self.correct
        self.correct_btn.pack()



    def correct(self):
        answer = elim_parenths(self.pred_values, 0)[0]
        self.display_text(answer)

        self.refit(self.pred_values)


    def incorrect(self):
        attributes = {"text", "incorrect_btn", "correct_btn", "e1", "adjust"}
        clear(self, attributes)

        self.e1 = tk.Entry(self)
        self.e1.pack()

        self.adjust = tk.Button(self)
        self.adjust["text"] = "Train"
        self.adjust["command"] = lambda: self.refit(self.e1.get().split())
        self.adjust.pack()

    def refit(self, correct_vals):

        if (len(correct_vals) != len(self.pred_values)):
            self.display_text("Error in input format, please try a different image")
        else:
            x = None
            y = []
            for i, val in enumerate(self.pred_values):
                corr_val = correct_vals[i]

                if x is None:
                    x = self.images[i]
                else:
                    x = combine_rows(x, self.images[i])

                for key in LABELS:
                    if LABELS[key] == corr_val:
                        y.append(key)



            self.model.compile()
            self.model.train(x, np.reshape(y, (len(y),)), epochs=50)
            self.model.save('model.json', "model.h5")
            self.extract_vals()


        attributes = {"e1", "adjust"}
        clear(self, attributes)


    def browse_file(self):
        self.img_path = filedialog.askopenfilename(initialdir="/", title="Select file",
                                   filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("jpeg", "*.jpeg"), ("png", "*.png")))
        self.show_image()



def elim_parenths(pred_values, start):
    stack = []
    i = start
    end = -1
    while i < len(pred_values):
        val = pred_values[i]

        if val == 14:
            x = elim_parenths(i + 1)
            stack.append(x[0])
            i = x[1]
            continue
        if val == 15:
            end = i
            break
        if val > 9:
            num = 0
            while len(stack) > 0 and isinstance(stack[-1], int):
                num *= 10
                num += stack[-1]
                stack.pop()

            stack.append(num)

            stack.append(LABELS[val])
        else:
            stack.append(val)

        i += 1

    m_d_rem = eval_pemdas(stack, {'*': lambda a,b: a * b, '/': lambda a,b: a / b})

    a_s_rem = eval_pemdas(m_d_rem, {'+': lambda a,b: a + b, '-': lambda a,b: a - b})

    return a_s_rem.pop(), end + 1

def eval_pemdas(stack, resolve_vals):
    i = 0
    while i < len(stack):
        val = stack[i]
        if val in resolve_vals:
            first = stack.pop(i - 1)
            stack.pop(i - 1)
            second = stack.pop(i - 1)

            stack.insert(i - 1, dict[val](first, second))
        else:
            i += 1
    return stack

def clear(obj, attributes):
    for attr in attributes:
        if hasattr(obj, attr):
            elem = getattr(obj, attr)
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
