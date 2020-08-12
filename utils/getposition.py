import wx
# https://stackoverflow.com/questions/57342753/how-to-select-a-rectangle-of-the-screen-to-capture-by-dragging-mouse-on-transpar

global selectionOffset, selectionSize

selectionOffset = ""
selectionSize = ""

class SelectableFrame(wx.Frame):

    c1 = None
    c2 = None

    def __init__(self, parent=None, id=wx.ID_ANY, title=""):
        wx.Frame.__init__(self, parent, id, title, size=wx.DisplaySize())
        self.menubar = wx.MenuBar(wx.MB_DOCKABLE)
        self.filem = wx.Menu()
        self.filem.Append(wx.ID_EXIT, '&Transparency')
        self.menubar.Append(self.filem, '&File')
        self.SetMenuBar(self.menubar)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_MENU, self.OnTrans)

        self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
        self.Show()
        self.transp = False
        wx.CallLater(250, self.OnTrans, None)

    def OnTrans(self, event):
        if self.transp == False:
            self.SetTransparent(180)
            self.transp = True
        else:
            self.SetTransparent(255)
            self.transp = False

    def OnMouseMove(self, event):
        if event.Dragging() and event.LeftIsDown():
            self.c2 = event.GetPosition()
            self.Refresh()

    def OnMouseDown(self, event):
        self.c1 = event.GetPosition()

    def OnMouseUp(self, event):
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
        self.Destroy()

    def OnPaint(self, event):
        global selectionOffset, selectionSize
        if self.c1 is None or self.c2 is None: return

        dc = wx.PaintDC(self)
        dc.SetPen(wx.Pen('red', 1))
        dc.SetBrush(wx.Brush(wx.Colour(0, 0, 0), wx.TRANSPARENT))

        dc.DrawRectangle(self.c1.x, self.c1.y, self.c2.x - self.c1.x, self.c2.y - self.c1.y)
        selectionOffset = str(self.c1.x) + "x" + str(self.c1.y)
        selectionSize = str(abs(self.c2.x - self.c1.x)) + "x" + str(abs(self.c2.y - self.c1.y))
    def PrintPosition(self, pos):
        return str(pos.x) + "x" + str(pos.y)


class MyApp(wx.App):

    def OnInit(self):
        frame = SelectableFrame()

        return True


app = MyApp(redirect=False)
app.MainLoop()
print('Display size: {}'.format(wx.DisplaySize()))
print("offset: " + selectionOffset + ". Screen selection size: " + selectionSize)