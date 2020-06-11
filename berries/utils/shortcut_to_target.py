# -*- coding: utf-8 -*-
# @Time   : 4/3/2020 4:25 AM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : shortcut_to_target.py

def shortcut_to_target(shortcut_path):
    import win32com.client
    return win32com.client.Dispatch("WScript.Shell").CreateShortCut(shortcut_path).Targetpath