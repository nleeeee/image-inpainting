#!/usr/bin/python

import os
import wx
from inpainting import inpaint

class InpaintingGUI(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(512,390))
        self.dirname = ''
        self.img = ''
        self.mask = ''
        self.CreateStatusBar()
        menubar = wx.MenuBar()
        file = wx.Menu()
        settings = wx.Menu()
        help = wx.Menu()
        
        imageOpen = wx.MenuItem(file, 101, '&Open Image\tCtrl+O', 'Open image')
        maskOpen = wx.MenuItem(file, 102, '&Open Mask\tCtrl+M', 'Open mask')
        quit = wx.MenuItem(file, 105, '&Quit\tCtrl+Q', 'Quit the Application')
        pref = wx.MenuItem(settings, 201, '&Settings', 'Change parameters')
        aboutInfo = wx.MenuItem(help, 301, '&About', '')
        
        file.AppendItem(imageOpen)
        file.AppendItem(maskOpen)
        file.AppendSeparator()
        file.AppendItem(quit)
        settings.AppendItem(pref)
        help.AppendItem(aboutInfo)
        
        menubar.Append(file, '&File')
        menubar.Append(settings, '&Settings')
        menubar.Append(help, '&Help')
        
        wx.Button(self, 1, 'Inpaint', (215, 290))
        wx.Button(self, 2, 'Load Image', (90, 250))
        wx.Button(self, 3, 'Load Mask', (340, 250))
        self.Bind(wx.EVT_BUTTON, self.onInpaint, id=1)
        self.Bind(wx.EVT_BUTTON, self.onOpenImage, id=2)
        self.Bind(wx.EVT_BUTTON, self.onOpenMask, id=3)
        self.Bind(wx.EVT_MENU, self.onOpenImage, id=101)
        self.Bind(wx.EVT_MENU, self.onOpenMask, id=102)
        self.Bind(wx.EVT_MENU, self.onAbout, id=301)
        self.Bind(wx.EVT_MENU, self.onQuit, id=105)
        
        self.SetMenuBar(menubar)
        self.Centre()
        self.Show()

    def onQuit(self, e):
        self.Close()
        
    def onAbout(self,e):
        # Create a message dialog box
        dlg = wx.MessageDialog(self, 'Image Inpainting', 'About', wx.OK)
        dlg.ShowModal() # Shows it
        dlg.Destroy() # finally destroy it when finished.
    
    def onOpenImage(self, e):
        '''Open a mask'''
        wildcard = 'JPEG files (*.jpg)|*.jpg|' +\
                   'PNG files (*.png)|*.png|' +\
                   'Other files (*.*)|*.*'
        dlg = wx.FileDialog(self, "Choose an image", wildcard=wildcard, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = open(os.path.join(self.dirname, self.filename), 'r')
            self.img = self.dirname + '/' + self.filename
        png = wx.Image(self.img, wx.BITMAP_TYPE_ANY)
        png = png.Scale(240, 240).ConvertToBitmap()
        wx.StaticBitmap(self, -1, png, (10, 5), (240, 240))
        dlg.Destroy()
        
    def onOpenMask(self, e):
        '''Open an image'''
        wildcard = 'BMP files (*.bmp)|*.bmp|' +\
                   'PGM files (*.pgm)|*.pgm|' +\
                   'Other files (*.*)|*.*'
        dlg = wx.FileDialog(self, "Choose a file", wildcard=wildcard, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = open(os.path.join(self.dirname, self.filename), 'r')
            self.mask = self.dirname + '/' + self.filename
        png = wx.Image(self.mask, wx.BITMAP_TYPE_ANY)
        png = png.Scale(240, 240).ConvertToBitmap()
        wx.StaticBitmap(self, -1, png, (260, 5), (240, 240))
        dlg.Destroy()
    
    def onInpaint(self, e):
        inpaint(self.img, self.mask)
        
if __name__ == '__main__':
    app = wx.App(False)
    frame = InpaintingGUI(None, -1, "Image Inpainting")
    app.MainLoop()
