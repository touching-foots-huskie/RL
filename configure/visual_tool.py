# Author: Harvey Chang
# Email: chnme40cs@gmail.com
import time
from functools import partial
import tkinter as tki
from tkinter import ttk
from collections import OrderedDict

global root, button_line, text_dict, refresh_dict, current_pane
text_dict = OrderedDict()
refresh_dict = OrderedDict()


def major_pane(config):
    root = tki.Tk()
    root.title('RL: Configuration')
    # create the menu frame
    menu_frame = ttk.Frame(root)
    menu_frame.pack(fill=tki.X, side=tki.TOP)
    menu_frame.tk_menuBar(help_menu(menu_frame))
    # info frame:
    info_frame = ttk.Frame(root)
    info_frame.pack(fill=tki.X, side=tki.BOTTOM)
    # first put the column label in a sub-frame:
    button_line = ttk.Frame(info_frame, relief=tki.RAISED, borderwidth=1)
    button_line.pack(side=tki.TOP, fill=tki.X, padx=1, pady=1)
    # fill in it:
    # two pane for config:
    # environment
    content_pane = ttk.Frame(info_frame, relief=tki.RAISED, borderwidth=1)
    content_pane.pack(side=tki.TOP, padx=1, pady=2, fill=tki.BOTH)

    # write all of sub_config:
    for name, sub_config in config.sub_config_dict.items():
        callback_func = partial(establish_sub_pane, content_pane=content_pane, sub_config=sub_config, config=config)
        refresh_dict[name] = callback_func
        ttk.Button(button_line, text=name, command=callback_func).pack(side=tki.LEFT)
    # add two button

    ttk.Button(button_line, text='done', command=partial(done, config=config)).pack(side=tki.LEFT)

    root.mainloop()


# establish
def help_menu(menu_frame):
    help_btn = ttk.Menubutton(menu_frame, text='Help', underline=0)
    help_btn.pack(side=tki.LEFT, padx='2m')
    help_btn.menu = tki.Menu(help_btn)
    help_btn.menu.add_command(label='How to', underline=0, command=how_to)
    help_btn['menu'] = help_btn.menu
    return help_btn


def establish_sub_pane(content_pane, sub_config, config, y_num=15):
    global current_pane
    clear_window(content_pane)
    for i, (key, value) in enumerate(sub_config.data.items()):
        establish_attribute_entry(content_pane, i, y_num, key, value, sub_config)
    config.refresh()
    current_pane = sub_config.name


def establish_attribute_entry(content_pane, id, y_num, key, value, sub_config):
    x, y = location_map(id, y_num)
    # put in frame:
    # ended with mark:
    if key[-4:] == 'mark':
        ttk.Label(content_pane, text=key, relief=tki.SOLID).grid(row=y, column=x, sticky='w', padx=2, pady=2)
        ttk.Label(content_pane, text=key, relief=tki.SOLID).grid(row=y, column=x+1, sticky='w', padx=2, pady=2)
        return

    ttk.Label(content_pane, text=key).grid(row=y, column=x, sticky='w', padx=2, pady=2)
    # value:
    text = tki.StringVar()
    text.set(value)
    text_dict[key] = text

    # entry of combo box:
    if key in sub_config.knowledge:
        combo = ttk.Combobox(content_pane, textvariable=text)
        combo['values'] = sub_config.knowledge[key]
        callback_func = partial(refresh_sub_config, sub_config=sub_config, name=key)
        combo.bind('<<ComboboxSelected>>', callback_func)
        combo.grid(row=y, column=x+1, sticky='w', padx=2, pady=2)
    else:
        entry = ttk.Entry(content_pane, textvariable=text)
        # bound call back
        callback_func = partial(refresh_sub_config, sub_config=sub_config, name=key)
        entry.bind('<Return>', callback_func)
        entry.grid(row=y, column=x+1, sticky='w', padx=2, pady=2)


# utils:
def location_map(iid, y_num):
    # for double align: using 2*x_id
    x_id = iid // y_num
    y_id = iid % y_num
    return 2*x_id, y_id


def get_time():
    result = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    return result


# callback functions
def how_to():
    print('BlaBla')


def clear_window(window):
    for widget in window.winfo_children():
        widget.destroy()


def refresh_sub_config(event, sub_config, name):
    value = text_dict[name].get()
    sub_config[name] = value
    sub_config.refresh(name)
    # refresh pane:
    refresh_dict[sub_config.name]()
    print(sub_config.data)


def done(config):
    global current_pane
    config.refresh()
    # print all:
    for key, value in config.data.items():
        print('{}:{}'.format(key, value))
    refresh_dict[current_pane]()
    config.save('train_log/{}'.format(get_time()))


if __name__ == '__main__':
    pass