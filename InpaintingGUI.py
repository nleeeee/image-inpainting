import os
import wx
from inpainting import inpaint

class InpaintingGUI(wx.Frame):
    
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, 
                          parent, 
                          id, 
                          title, 
                          wx.DefaultPosition, 
                          wx.Size(608,290))
                          
        self.dirname = ''
        self.img = ''
        self.mask = ''
        self.patch_size = 9 # default patch size
        self.gauss = 0 # no Gaussian smoothing by default
        self.sigma = 1 # sigma value for Gaussian smoothing
        
        txtImage = 'Image'
        txtMask = 'Mask'
        labelImage = wx.StaticText(self, label=txtImage, pos=(106,10))
        labelMask = wx.StaticText(self, label=txtMask, pos=(356,10))
        
        self.CreateStatusBar()
        menubar = wx.MenuBar()
        file = wx.Menu()
        settings = wx.Menu()
        help = wx.Menu()
        
        imageOpen = wx.MenuItem(file, 101, '&Open Image\tCtrl+O', 'Open image')
        maskOpen = wx.MenuItem(file, 102, '&Open Mask\tCtrl+M', 'Open mask')
        quit = wx.MenuItem(file, 105, '&Quit\tCtrl+Q', 'Quit the Application')
        patchSize = wx.MenuItem(settings, 201, '&Patch Size', 'Change patch size')
        gauss = wx.MenuItem(settings, 202, '&Gaussian Smoothing', '')
        aboutInfo = wx.MenuItem(help, 301, '&About', '')
        
        file.AppendItem(imageOpen)
        file.AppendItem(maskOpen)
        file.AppendSeparator()
        file.AppendItem(quit)
        settings.AppendItem(patchSize)
        settings.AppendItem(gauss)
        help.AppendItem(aboutInfo)
        
        menubar.Append(file, '&File')
        menubar.Append(settings, '&Settings')
        menubar.Append(help, '&Help')
        
        wx.Button(self, 1, 'Inpaint', (10, 100))
        wx.Button(self, 2, 'Load Image', (10, 10))
        wx.Button(self, 3, 'Load Mask', (10, 40))
        self.Bind(wx.EVT_BUTTON, self.onInpaint, id=1)
        self.Bind(wx.EVT_BUTTON, self.onOpenImage, id=2)
        self.Bind(wx.EVT_BUTTON, self.onOpenMask, id=3)
        self.Bind(wx.EVT_MENU, self.onOpenImage, id=101)
        self.Bind(wx.EVT_MENU, self.onOpenMask, id=102)
        self.Bind(wx.EVT_MENU, self.onAbout, id=301)
        self.Bind(wx.EVT_MENU, self.onPatchSize, id=201)
        #self.Bind(wx.EVT_MENU, self.onGauss, id=202)
        self.Bind(wx.EVT_MENU, self.onQuit, id=105)
        
        self.SetMenuBar(menubar)
        self.Centre()
        self.Show()

    def onQuit(self, e):
        '''Closes GUI'''
        self.Close()
        
    def onAbout(self,e):
        '''Opens About dialog'''
        dlg = wx.MessageDialog(self, 'Image Inpainting', 'About', wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
    
    def onOpenImage(self, e):
        '''Open image to inpaint'''
        wildcard = 'JPEG files (*.jpg)|*.jpg|' +\
                   'PNG files (*.png)|*.png|' +\
                   'Other files (*.*)|*.*'
        dlg = wx.FileDialog(self, "Choose an image", 
                            wildcard=wildcard, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = open(os.path.join(self.dirname, self.filename), 'r')
            self.img = self.dirname + '/' + self.filename
        png = wx.Image(self.img, wx.BITMAP_TYPE_ANY)
        png = png.Scale(240, 240).ConvertToBitmap()
        wx.StaticBitmap(self, -1, png, (106, 25), (240, 240))
        dlg.Destroy()
        
    def onOpenMask(self, e):
        '''Open mask'''
        wildcard = 'BMP files (*.bmp)|*.bmp|' +\
                   'PGM files (*.pgm)|*.pgm|' +\
                   'Other files (*.*)|*.*'
        dlg = wx.FileDialog(self, "Choose a file", 
                            wildcard=wildcard, style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            f = open(os.path.join(self.dirname, self.filename), 'r')
            self.mask = self.dirname + '/' + self.filename
        png = wx.Image(self.mask, wx.BITMAP_TYPE_ANY)
        png = png.Scale(240, 240).ConvertToBitmap()
        wx.StaticBitmap(self, -1, png, (356, 25), (240, 240))
        dlg.Destroy()
        
    def onPatchSize(self, e):
        '''Dialog to set patch size'''
        ps = 'Enter patch size value:'
        dlg = wx.NumberEntryDialog(self, '', ps, 'Patch Size', 
                                   self.patch_size, 1, 1000)
        if dlg.ShowModal() == wx.ID_OK:
            self.patch_size = dlg.GetValue()
        dlg.Destroy()
        
    def onGauss(self, e):
        '''Configures to wheter apply Gaussian smoothing to the image 
        prior to calculating iamge gradients.'''
        return
    
    def onInpaint(self, e):
        '''Runs the inpainting algorithm when Inpaint button is clicked'''
        inpaint(self.img, self.mask, self.gauss, self.sigma, self.patch_size)
        
if __name__ == '__main__':
    app = wx.App(False)
    frame = InpaintingGUI(None, -1, "Image Inpainting")
    app.MainLoop()