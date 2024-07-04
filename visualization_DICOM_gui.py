
# --------------------------------------------------
#
#     Copyright (C) 2024


import numpy as np

import qtpy.QtCore as QC
import qtpy.QtWidgets as QW
import qtpy.QtGui as QG

import guidata.dataset.datatypes as gdt
import guidata.dataset.dataitems as gdi
import guidata.dataset.qtwidgets as gdq
import guidata.configtools as configtools

import guiqwt.plot as gqp
import guiqwt.curve as gqc
import guiqwt.image as gqi
import guiqwt.tools as gqt


import os.path as osp

from guiqwt.plot import ImageDialog
from guiqwt.tools import (
    RectangleTool,
    EllipseTool,
    HRangeTool,
    PlaceAxesTool,
    MultiLineTool,
    FreeFormTool,
    SegmentTool,
    CircleTool,
    AnnotatedRectangleTool,
    AnnotatedEllipseTool,
    AnnotatedSegmentTool,
    AnnotatedCircleTool,
    LabelTool,
    AnnotatedPointTool,
    VCursorTool,
    HCursorTool,
    XCursorTool,
    ObliqueRectangleTool,
    AnnotatedObliqueRectangleTool,
)
from guiqwt.builder import make


def create_window():
    win = ImageDialog(
        edit=False, toolbar=True, wintitle="All image and plot tools test"
    )
    for toolklass in (
        LabelTool,
        HRangeTool,
        VCursorTool,
        HCursorTool,
        XCursorTool,
        SegmentTool,
        RectangleTool,
        ObliqueRectangleTool,
        CircleTool,
        EllipseTool,
        MultiLineTool,
        FreeFormTool,
        PlaceAxesTool,
        AnnotatedRectangleTool,
        AnnotatedObliqueRectangleTool,
        AnnotatedCircleTool,
        AnnotatedEllipseTool,
        AnnotatedSegmentTool,
        AnnotatedPointTool,
    ):
        win.add_tool(toolklass)
    return win


def test(filname_p=None):
    """Test"""
    # -- Create QApplication
    import guidata

    _app = guidata.qapplication()
    # --
    # filename = osp.join(osp.dirname(__file__), "1.dcm")
    filename = osp.join(filname_p)
    win = create_window()
    image = make.image(filename=filename, colormap="bone")
    plot = win.get_plot()
    plot.add_item(image)
    win.exec_()
import tkinter as tk
from tkinter import filedialog
import os

application_window = tk.Tk()

# Build a list of tuples for each file type the file dialog should display
my_filetypes = [('all files', '.*'), ('text files', '.txt')]

# plots = []
#
# for f in glob.glob("/home/kevinbk/PycharmProjects/pythonProjectforDICOM/DICM/*.dcm"):
#     pass
#     filename = f.split("/")[-1]
#     ds = pydicom.dcmread("1.dcm")
#     pix = ds.pixel_array
#     pix = pix * 1 + (-1024)
#     plots.append(pix)
#
# y = np.dstack(plots)
#
# tracker = Dicom_data(ax, y)
#
#
# plt.show()

# print(pix.shape)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# Z = pix
#
# # image rotation
# do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))
#
# # image skew
# do_plot(ax2, Z, mtransforms.Affine2D().skew_deg(30, 15))
#
# # scale and reflection
# do_plot(ax3, Z, mtransforms.Affine2D().scale(-1, .5))
#
# # everything and a translation
# do_plot(ax4, Z, mtransforms.Affine2D().
#         rotate_deg(30).skew_deg(30, 15).scale(-1, .5).translate(.5, -1))
#
# plt.show()

# pix = pix * 1 + (-1024)
# print(pix.shape)
# trans = transforms.Compose([transforms.ToTensor()])
#
# # demo = Image.open(img)
# demo_img = trans(pix)
# demo_array = np.moveaxis(demo_img.numpy() * 255, 0, -1)

if __name__ == "__main__":

    answer = filedialog.askopenfilename(parent=application_window,
                                        initialdir=os.getcwd(),
                                        title="Please select a file:",
                                        filetypes=my_filetypes)
    application_window.destroy()
    print(answer)

    test(filname_p=answer)
